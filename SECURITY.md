# Security

Report vulnerabilities using the repository's private security-advisory form.
Provide a synthetic reproduction and describe whether the issue affects Python
callable loading, subprocess evaluation, checkpoint parsing, or output paths.

Only load evaluators that you trust. A Python evaluator is imported into the
optimizer process and therefore has that process's permissions. The command
adapter passes an argv array directly to the operating system and never enables
a shell, but the selected executable is still trusted code.

The newest minor release receives security fixes.
