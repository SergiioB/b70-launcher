# Security Policy

## Supported Versions

Only the latest release is supported. Launcher updates and recipe updates are
delivered through the xecores.com release channel; run the newest version.

## Reporting a Vulnerability

Please report vulnerabilities privately via **GitHub Security Advisories**
(“Report a vulnerability” on the Security tab of this repository) rather than
public issues. Include reproduction steps and, if relevant, the launcher
version and platform.

The launcher runs a local HTTP API on loopback. If you find an auth bypass,
path traversal, command injection, or recipe-overlay escape, that is a
security bug — please report it before disclosure.
