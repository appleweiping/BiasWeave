"""Transactional, race-safe output shared by every BiasWeave writer."""

from __future__ import annotations

import os
import secrets
import tempfile
import unicodedata
from collections.abc import Iterable, Sequence
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import TracebackType

from biasweave.errors import CheckpointError


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _Installed:
    target: Path
    identity: _FileIdentity


@dataclass(frozen=True, slots=True)
class _Backup:
    path: Path
    target: Path
    identity: _FileIdentity


def _file_identity(path: Path) -> _FileIdentity:
    details = path.stat(follow_symlinks=False)
    return _FileIdentity(details.st_dev, details.st_ino)


def _same_file_identity(path: Path, identity: _FileIdentity) -> bool:
    try:
        return _file_identity(path) == identity
    except FileNotFoundError:
        return False


class WriterClaim:
    """Fail-closed, nonce-owned single-writer claim for one directory scope."""

    def __init__(self, root: str | Path, *, scope: str = "run") -> None:
        self.root = Path(root)
        try:
            normalized_scope = unicodedata.normalize("NFC", scope)
            encoded_scope = normalized_scope.encode("utf-8")
        except (TypeError, UnicodeEncodeError) as error:
            raise CheckpointError(f"writer claim scope is not valid Unicode: {error}") from error
        self.scope = normalized_scope
        digest = sha256(encoded_scope).hexdigest()[:32]
        self.directory = self.root / f".biasweave-writer-{digest}"
        self.owner = self.directory / "owner"
        self.token = secrets.token_hex(32).encode("ascii")
        self._directory_identity: _FileIdentity | None = None
        self._owner_identity: _FileIdentity | None = None
        self._active = False

    def __enter__(self) -> WriterClaim:
        created = False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.root.is_symlink() or not self.root.is_dir():
                raise CheckpointError(f"writer claim root is not a regular directory: {self.root}")
            try:
                self.directory.mkdir(mode=0o700)
                created = True
            except FileExistsError as error:
                raise CheckpointError(
                    f"another writer holds the output claim: {self.directory}"
                ) from error
            if (
                self.directory.is_symlink()
                or not self.directory.is_dir()
                or self.directory.resolve() != self.root.resolve() / self.directory.name
            ):
                raise CheckpointError(
                    f"writer claim path is redirected or invalid: {self.directory}"
                )
            self._directory_identity = _file_identity(self.directory)
            with self.owner.open("xb") as stream:
                owner_details = os.fstat(stream.fileno())
                self._owner_identity = _FileIdentity(owner_details.st_dev, owner_details.st_ino)
                view = memoryview(self.token)
                while view:
                    written = stream.write(view)
                    if written <= 0:
                        raise OSError("writer claim made no progress")
                    view = view[written:]
                stream.flush()
                os.fsync(stream.fileno())
            self._active = True
            return self
        except CheckpointError:
            if created:
                self._remove_incomplete_claim()
            raise
        except OSError as error:
            if created:
                self._remove_incomplete_claim()
            raise CheckpointError(f"cannot acquire output writer claim: {error}") from error
        except BaseException:
            if created:
                self._remove_incomplete_claim()
            raise

    def _remove_incomplete_claim(self) -> None:
        try:
            if self._directory_identity is None or not _same_file_identity(
                self.directory, self._directory_identity
            ):
                return
            if self._owner_identity is not None and _same_file_identity(
                self.owner, self._owner_identity
            ):
                self.owner.unlink()
            self.directory.rmdir()
        except OSError:
            # Never recursively delete an uncertain claim. A leftover claim is
            # an intentional fail-closed signal requiring operator inspection.
            pass

    def owns(self, root: str | Path, *, scope: str | None = None) -> bool:
        """Return whether this live claim coordinates the requested root and scope."""

        try:
            if not self._active or Path(root).resolve() != self.root.resolve():
                return False
            if scope is not None and unicodedata.normalize("NFC", scope) != self.scope:
                return False
            if self._directory_identity is None or self._owner_identity is None:
                return False
            return (
                _same_file_identity(self.directory, self._directory_identity)
                and _same_file_identity(self.owner, self._owner_identity)
                and self._owner_payload() == self.token
            )
        except (OSError, RuntimeError, TypeError):
            return False

    def _owner_payload(self) -> bytes:
        with self.owner.open("rb") as stream:
            return stream.read(len(self.token) + 1)

    def _release(self) -> None:
        if not self._active:
            return
        if not self.owns(self.root):
            self._active = False
            raise CheckpointError(
                f"writer claim ownership changed; refusing to remove it: {self.directory}"
            )
        try:
            self.owner.unlink()
            self.directory.rmdir()
        except OSError as error:
            self._active = False
            raise CheckpointError(f"cannot release output writer claim: {error}") from error
        except BaseException as error:
            # If cancellation arrived after our owner unlink completed, the
            # still-identical empty directory is safe to finish removing. If
            # anything appeared in it, rmdir fails and leaves the marker
            # fail-closed for inspection.
            try:
                if (
                    self._directory_identity is not None
                    and _same_file_identity(self.directory, self._directory_identity)
                    and not os.path.lexists(self.owner)
                ):
                    self.directory.rmdir()
            except OSError as cleanup_error:
                error.add_note(f"writer claim cleanup did not complete: {cleanup_error}")
            self._active = False
            raise
        self._active = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        try:
            self._release()
        except CheckpointError as error:
            if exc_value is None:
                raise
            exc_value.add_note(str(error))


def _identity(path: Path) -> str:
    return unicodedata.normalize("NFC", str(path.resolve(strict=False))).casefold()


def paths_alias(first: str | Path, second: str | Path) -> bool:
    """Recognize lexical, case-folded, symlink, and existing hard-link aliases."""

    left = Path(first)
    right = Path(second)
    try:
        return _identity(left) == _identity(right) or (
            left.exists() and right.exists() and left.samefile(right)
        )
    except OSError as error:
        raise CheckpointError(f"cannot resolve output path alias: {error}") from error


def preflight_outputs(
    destinations: Iterable[str | Path],
    *,
    force: bool,
    protected: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Validate a complete output set before creating or replacing any file."""

    targets = tuple(Path(destination) for destination in destinations)
    protected_paths = tuple(Path(path) for path in protected if str(path))
    for index, target in enumerate(targets):
        if any(paths_alias(target, source) for source in protected_paths):
            raise CheckpointError(f"output {target} aliases an input and is never writable")
        if any(paths_alias(target, prior) for prior in targets[:index]):
            raise CheckpointError(f"output destinations alias each other: {target}")
        if target.exists() and target.is_dir():
            raise CheckpointError(f"output destination is a directory: {target}")
        if target.exists() and not force:
            raise CheckpointError(f"refusing to overwrite existing output: {target}; pass --force")
    return targets


def protect_open_descriptor(
    descriptor: int,
    target: str | Path,
    protected: Iterable[str | Path],
) -> None:
    """Reject a protected hard-link/symlink race after an append handle opens."""

    try:
        opened = os.fstat(descriptor)
        opened_identity = (opened.st_dev, opened.st_ino)
        for source in protected:
            source_path = Path(source)
            try:
                source_stat = source_path.stat()
            except FileNotFoundError:
                continue
            if opened_identity == (source_stat.st_dev, source_stat.st_ino):
                raise CheckpointError(
                    f"output {Path(target)} aliases an input and is never writable"
                )
    except CheckpointError:
        raise
    except OSError as error:
        raise CheckpointError(f"cannot verify open output path {Path(target)}: {error}") from error


def atomic_write_many(
    outputs: Sequence[tuple[str | Path, bytes]],
    *,
    force: bool = False,
    protected: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Stage a set and roll back every failure before the explicit commit point."""

    targets = preflight_outputs(
        (destination for destination, _payload in outputs),
        force=force,
        protected=protected,
    )
    if not targets:
        return targets
    staged: list[tuple[Path, Path]] = []
    backups: list[_Backup] = []
    installed: list[_Installed] = []
    with ExitStack() as claims:
        claim_roots = sorted(
            {(target.parent.resolve(), _identity(target)) for target in targets},
            key=lambda item: (str(item[0]).casefold(), item[1]),
        )
        for parent, scope in claim_roots:
            claims.enter_context(WriterClaim(parent, scope=f"output:{scope}"))
        # Existence and aliases are checked again only after every claim is
        # held. This closes the gap between the public preflight and staging.
        preflight_outputs(targets, force=force, protected=protected)
        try:
            for target, (_destination, payload) in zip(targets, outputs, strict=True):
                if not isinstance(payload, bytes):
                    raise CheckpointError("atomic output payloads must be bytes")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    dir=target.parent,
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    staged.append((temporary, target))
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())

            if force:
                for _temporary, target in staged:
                    if not os.path.lexists(target):
                        continue
                    original_identity = _file_identity(target)
                    descriptor, backup_name = tempfile.mkstemp(
                        prefix=f".{target.name}.", suffix=".bak", dir=target.parent
                    )
                    os.close(descriptor)
                    backup = Path(backup_name)
                    backup.unlink()
                    backups.append(_Backup(backup, target, original_identity))
                    os.replace(target, backup)
                    if not _same_file_identity(backup, original_identity):
                        raise CheckpointError(
                            f"output identity changed while being backed up: {target}"
                        )
                    if os.path.lexists(target):
                        raise CheckpointError(
                            f"concurrent output appeared while backing up: {target}"
                        )
                for temporary, target in staged:
                    installed_identity = _file_identity(temporary)
                    installed.append(_Installed(target, installed_identity))
                    try:
                        # The backup phase made this name absent. A hard-link
                        # install is no-clobber, closing the remaining
                        # check/install race even for an uncoordinated writer.
                        os.link(temporary, target)
                    except FileExistsError as error:
                        raise CheckpointError(
                            f"concurrent output appeared during forced install: {target}"
                        ) from error
                    if not _same_file_identity(target, installed_identity):
                        raise CheckpointError(f"installed output identity changed: {target}")
            else:
                for temporary, target in staged:
                    installed_identity = _file_identity(temporary)
                    installed.append(_Installed(target, installed_identity))
                    try:
                        os.link(temporary, target)
                    except FileExistsError as error:
                        raise CheckpointError(
                            f"refusing to overwrite existing output: {target}; pass --force"
                        ) from error
                    if not _same_file_identity(target, installed_identity):
                        raise CheckpointError(f"installed output identity changed: {target}")
        except BaseException as error:
            rollback_errors = _rollback(installed, backups)
            if rollback_errors:
                error.add_note(f"output rollback errors: {rollback_errors}")
            if isinstance(error, OSError) and not isinstance(error, CheckpointError):
                suffix = f"; rollback errors: {rollback_errors}" if rollback_errors else ""
                raise CheckpointError(
                    f"cannot install output transaction: {error}{suffix}"
                ) from error
            raise
        finally:
            for temporary, _target in staged:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

        # Every new identity is now installed: this is the transaction's
        # linearization/commit point. Backup cleanup cannot be described as a
        # rollback failure after this point because some old identities may
        # already have been deleted.
        try:
            cleanup_errors = _discard_backups(backups)
        except BaseException as error:
            error.add_note("output transaction committed before backup cleanup was interrupted")
            raise
        if cleanup_errors:
            raise CheckpointError(
                f"output transaction committed but cannot remove output backups: {cleanup_errors}"
            )
    return targets


def _rollback(installed: list[_Installed], backups: list[_Backup]) -> str:
    failures: list[str] = []
    backed_up_targets = {backup.target for backup in backups}
    installed_by_target = {entry.target: entry for entry in installed}
    for entry in reversed(installed):
        try:
            if entry.target not in backed_up_targets and _same_file_identity(
                entry.target, entry.identity
            ):
                entry.target.unlink()
        except OSError as error:
            failures.append(str(error))
    for backup in reversed(backups):
        try:
            if not _same_file_identity(backup.path, backup.identity):
                if _same_file_identity(backup.target, backup.identity):
                    # The backup move failed before changing either path.
                    if os.path.lexists(backup.path):
                        failures.append(f"foreign backup path preserved: {backup.path}")
                    continue
                foreign_error = _restore_foreign_backup(backup)
                if foreign_error:
                    failures.append(foreign_error)
                continue
            installed_entry = installed_by_target.get(backup.target)
            target_is_ours = installed_entry is not None and _same_file_identity(
                backup.target, installed_entry.identity
            )
            if target_is_ours:
                os.replace(backup.path, backup.target)
            elif not os.path.lexists(backup.target):
                try:
                    os.link(backup.path, backup.target)
                except FileExistsError:
                    failures.append(f"concurrent output blocks backup restore: {backup.target}")
                    continue
                backup.path.unlink()
            else:
                failures.append(f"concurrent output blocks backup restore: {backup.target}")
        except OSError as error:
            failures.append(str(error))
    return "; ".join(failures)


def _restore_foreign_backup(backup: _Backup) -> str:
    """Put a regular concurrent replacement back without clobbering a newer one."""

    try:
        identity = _file_identity(backup.path)
        if os.path.lexists(backup.target):
            return f"concurrent output blocks displaced replacement restore: {backup.target}"
        if backup.path.is_symlink() or not backup.path.is_file():
            return f"displaced non-regular replacement preserved at: {backup.path}"
        try:
            os.link(backup.path, backup.target)
        except FileExistsError:
            return f"concurrent output blocks displaced replacement restore: {backup.target}"
        if not _same_file_identity(backup.target, identity):
            return f"displaced replacement identity changed during restore: {backup.target}"
        if not _same_file_identity(backup.path, identity):
            return f"displaced replacement backup changed during restore: {backup.path}"
        backup.path.unlink()
        return ""
    except OSError as error:
        return str(error)


def _discard_backups(backups: list[_Backup]) -> str:
    failures: list[str] = []
    for backup in backups:
        try:
            if _same_file_identity(backup.path, backup.identity):
                backup.path.unlink()
            elif os.path.lexists(backup.path):
                failures.append(f"backup ownership changed: {backup.path}")
        except OSError as error:
            failures.append(str(error))
    return "; ".join(failures)


def atomic_write_bytes(
    destination: str | Path,
    payload: bytes,
    *,
    force: bool = False,
    protected: Iterable[str | Path] = (),
) -> Path:
    return atomic_write_many([(destination, payload)], force=force, protected=protected)[0]


def utf8(text: str) -> bytes:
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CheckpointError(f"output text is not valid Unicode: {error}") from error
