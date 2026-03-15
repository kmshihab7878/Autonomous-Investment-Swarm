# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in AIS, please report it responsibly:

1. **Preferred**: Use [GitHub Security Advisories](https://github.com/kmshihab7878/Financial-Intelligence-Department-FID/security/advisories/new) to report the issue privately.
2. **Fallback**: Email khaledshihab73@gmail.com with the subject line `[AIS Security]`.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Fix or mitigation**: Best effort, depending on severity

## Scope

The following are considered security issues in this trading system:

- Unauthorized order submission or state mutation
- Credential exposure (API keys, HMAC secrets, database passwords)
- Injection vulnerabilities in API endpoints
- Privilege escalation in session or mandate controls
- Safety gate circumvention

## Out of Scope

- Vulnerabilities in third-party dependencies (report upstream)
- Issues requiring physical access to the deployment host
- Social engineering attacks
- Trading strategy effectiveness or financial losses from legitimate use

## Security Design

AIS is built with several security controls:

- **HMAC-signed risk tokens**: Every order requires a cryptographically signed approval from the risk engine
- **Fail-closed defaults**: Missing secrets cause immediate startup failure
- **Paper mode default**: Live trading requires explicit opt-in via `AIS_ENABLE_LIVE_TRADING=true`
- **Timing-attack prevention**: Constant-time comparison for token validation
- **API authentication**: Bearer token required for all control plane endpoints
