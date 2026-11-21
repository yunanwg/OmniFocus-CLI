# HTTPS API Reference

`omnifocus-cli` can run a private HTTPS JSON API for n8n and other automation clients.

This transport is separate from MCP. It is designed for machine-to-machine access and uses
authenticated OpenAPI as its canonical contract.

## Security Model

The HTTPS API is intentionally strict:

- TLS is mandatory and enforced at **TLS 1.3 minimum**
- Bearer authentication is mandatory on **every** endpoint
- the server refuses to start without:
  - `OF_HTTP_API_KEY`
  - `OF_HTTP_TLS_CERT_FILE`
  - `OF_HTTP_TLS_KEY_FILE`
- default bind address is `127.0.0.1`
- default port is `8443`
- no Swagger UI or Redoc is exposed in production mode
- the OpenAPI spec at `/v1/openapi.json` is also protected by Bearer auth
- request bodies are capped at 1 MiB
- invalid Bearer attempts are rate-limited in-memory
- trusted hosts are allowlisted

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `OF_HTTP_HOST` | No | Bind host. Defaults to `127.0.0.1`. |
| `OF_HTTP_PORT` | No | Bind port. Defaults to `8443`. |
| `OF_HTTP_API_KEY` | Yes | Required Bearer token value. |
| `OF_HTTP_API_KEY_FILE` | No | File containing the Bearer token; conflicts with `OF_HTTP_API_KEY`. |
| `OF_HTTP_TLS_CERT_FILE` | Yes | Path to the TLS certificate PEM file. |
| `OF_HTTP_TLS_KEY_FILE` | Yes | Path to the TLS private key PEM file. |
| `OF_HTTP_ALLOWED_HOSTS` | No | Comma-separated trusted host allowlist. Defaults to `127.0.0.1,localhost`. |
| `OF_WEBDAV_URL` | Usually | Required for endpoints that need bundle access. |
| `OF_WEBDAV_USER` | No | Explicit WebDAV username override. |
| `OF_WEBDAV_PASS` | No | Explicit WebDAV password override. |
| `OF_ENCRYPTION_PASSPHRASE` | No | Bundle decryption passphrase. |
| `OF_CACHE_DIR` | No | Writable cache directory. |

## Copy-paste localhost TLS and Hermes MCP setup

This creates a self-signed server certificate that is trusted only by Hermes. It has Subject
Alternative Names for both `localhost` and `127.0.0.1`; use `https://localhost:8443/mcp/` in the
Hermes configuration so normal hostname validation succeeds. This is preferable to disabling
verification or pinning a certificate fingerprint.

Run once on the host that runs both Podman and Hermes:

```bash
umask 077
TLS_DIR="$HOME/.config/omnifocus-mcp/tls"
mkdir -p "$TLS_DIR"
mkdir -p "$HOME/.local/share/omnifocus-mcp-cache"
chmod 700 "$HOME/.local/share/omnifocus-mcp-cache"

openssl genpkey -algorithm ED25519 -out "$TLS_DIR/server-key.pem"
openssl req -x509 -new -sha256 -days 397 \
  -key "$TLS_DIR/server-key.pem" \
  -out "$TLS_DIR/server-cert.pem" \
  -subj '/CN=localhost' \
  -addext 'subjectAltName=DNS:localhost,IP:127.0.0.1' \
  -addext 'basicConstraints=critical,CA:FALSE' \
  -addext 'keyUsage=critical,digitalSignature' \
  -addext 'extendedKeyUsage=serverAuth'
openssl rand -hex 32 > "$TLS_DIR/api-token"
chmod 600 "$TLS_DIR/server-key.pem" "$TLS_DIR/api-token"

podman secret create of-http-api-key "$TLS_DIR/api-token"
```

Create the WebDAV secrets as described in [Container and Runtime Notes](container.md#secrets),
then start one long-lived local service:

```bash
podman run --rm -d --name omnifocus-mcp \
  --userns=keep-id:uid=1001,gid=1001 \
  -p 127.0.0.1:8443:8443 \
  -v "$TLS_DIR":/tls:ro \
  -v "$HOME/.local/share/omnifocus-mcp-cache":/cache \
  --secret of-http-api-key,target=of-http-api-key,uid=1001,gid=1001,mode=0400 \
  --secret of-webdav-url,target=of-webdav-url,uid=1001,gid=1001,mode=0400 \
  --secret of-webdav-user,target=of-webdav-user,uid=1001,gid=1001,mode=0400 \
  --secret of-webdav-pass,target=of-webdav-pass,uid=1001,gid=1001,mode=0400 \
  -e OF_HTTP_HOST=0.0.0.0 \
  -e OF_CACHE_DIR=/cache \
  -e OF_HTTP_API_KEY_FILE=/run/secrets/of-http-api-key \
  -e OF_HTTP_TLS_CERT_FILE=/tls/server-cert.pem \
  -e OF_HTTP_TLS_KEY_FILE=/tls/server-key.pem \
  -e OF_WEBDAV_URL_FILE=/run/secrets/of-webdav-url \
  -e OF_WEBDAV_USER_FILE=/run/secrets/of-webdav-user \
  -e OF_WEBDAV_PASS_FILE=/run/secrets/of-webdav-pass \
  ghcr.io/szymczag/omnifocus-cli:v1.4.1 http
```

`OF_HTTP_HOST=0.0.0.0` binds the server to the container network interface so rootless Podman can
forward the published port. It does not make the service remotely reachable: the `PublishPort`
equivalent above still exposes it only as `127.0.0.1:8443` on the host. The user namespace mapping
lets the non-root image user read the `0600` key and certificate files without relaxing their host
permissions.

### Start automatically with rootless systemd

After confirming the manual container works, replace it with a rootless Quadlet. This starts the
service on login, restarts it after failures, and (with lingering enabled) starts it after host
reboots even before you log in.

```bash
podman stop omnifocus-mcp
mkdir -p "$HOME/.config/containers/systemd"
```

Create `~/.config/containers/systemd/omnifocus-mcp.container` with the following content:

```ini
[Unit]
Description=OmniFocus HTTPS MCP service
Wants=network-online.target
After=network-online.target

[Container]
Image=ghcr.io/szymczag/omnifocus-cli:v1.4.1
ContainerName=omnifocus-mcp
UserNS=keep-id:uid=1001,gid=1001
PublishPort=127.0.0.1:8443:8443
Volume=%h/.config/omnifocus-mcp/tls:/tls:ro
Volume=%h/.local/share/omnifocus-mcp-cache:/cache
Secret=of-http-api-key,target=of-http-api-key,uid=1001,gid=1001,mode=0400
Secret=of-webdav-url,target=of-webdav-url,uid=1001,gid=1001,mode=0400
Secret=of-webdav-user,target=of-webdav-user,uid=1001,gid=1001,mode=0400
Secret=of-webdav-pass,target=of-webdav-pass,uid=1001,gid=1001,mode=0400
Environment=OF_HTTP_HOST=0.0.0.0
Environment=OF_CACHE_DIR=/cache
Environment=OF_HTTP_API_KEY_FILE=/run/secrets/of-http-api-key
Environment=OF_HTTP_TLS_CERT_FILE=/tls/server-cert.pem
Environment=OF_HTTP_TLS_KEY_FILE=/tls/server-key.pem
Environment=OF_WEBDAV_URL_FILE=/run/secrets/of-webdav-url
Environment=OF_WEBDAV_USER_FILE=/run/secrets/of-webdav-user
Environment=OF_WEBDAV_PASS_FILE=/run/secrets/of-webdav-pass
Exec=http

[Service]
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

Then enable it and keep the user service manager running across reboots:

```bash
systemctl --user daemon-reload
systemctl --user start omnifocus-mcp.service
sudo loginctl enable-linger "$USER"
```

Quadlet services are generated units, so `systemctl --user enable` intentionally does not work for
them. Quadlet applies the `[Install]` section during generation; the explicit `start` command
starts the service immediately. Enabling lingering needs administrator authorization and makes the
user systemd manager start at boot, so the generated `WantedBy=default.target` dependency starts
the service without a login session.

Verify the service with `systemctl --user status omnifocus-mcp.service` and inspect logs with
`journalctl --user -u omnifocus-mcp.service`. Keep `PublishPort=127.0.0.1:8443:8443`; changing it
to `0.0.0.0:8443:8443` would expose the full read/write MCP service to the network.

Add the following token reference to `~/.hermes/.env` without putting the token in
`config.yaml`:

```bash
printf '\nOMNIFOCUS_MCP_TOKEN=%s\n' "$(<"$TLS_DIR/api-token")" >> "$HOME/.hermes/.env"
chmod 600 "$HOME/.hermes/.env"
```

Then add this to `~/.hermes/config.yaml` and reload MCP configuration in Hermes:

```yaml
mcp_servers:
  omnifocus:
    url: https://localhost:8443/mcp/
    ssl_verify: ~/.config/omnifocus-mcp/tls/server-cert.pem
    headers:
      Authorization: Bearer ${OMNIFOCUS_MCP_TOKEN}
    tools:
      resources: false
      prompts: false
```

The `/mcp/` endpoint exposes the complete OmniFocus MCP tool set, including tools that create,
update, complete, drop, and sync OmniFocus data. The same Bearer token is required for every
request, including MCP initialization, tool discovery, and mutations. Keep the service bound to
`127.0.0.1`; do not use this self-signed setup for a remotely reachable service. To rotate the
certificate or token, recreate the relevant file/Podman secret, recreate the container, and reload
Hermes.

## Start the Server

### Native Python entrypoint

```bash
OF_HTTP_API_KEY=replace-me \
OF_HTTP_TLS_CERT_FILE=/absolute/path/cert.pem \
OF_HTTP_TLS_KEY_FILE=/absolute/path/key.pem \
of-http
```

### Container launcher mode

```bash
podman run --rm \
  -p 127.0.0.1:8443:8443 \
  -v "$PWD/certs":/tls:ro \
  -e OF_HTTP_API_KEY=replace-me \
  -e OF_HTTP_TLS_CERT_FILE=/tls/cert.pem \
  -e OF_HTTP_TLS_KEY_FILE=/tls/key.pem \
  ghcr.io/szymczag/omnifocus-cli:latest http
```

## OpenAPI

The canonical machine-readable schema lives at:

- `GET /v1/openapi.json`

It is protected by the same Bearer auth path as the rest of the API.
That OpenAPI document is the normative HTTP contract; this Markdown page is an operator guide
and implementation summary.

Example:

```bash
curl --silent --show-error --cacert /absolute/path/to/server-cert.pem \
  -H "Authorization: Bearer replace-me" \
  https://127.0.0.1:8443/v1/openapi.json
```

## n8n Usage

Use the standard HTTP Request node:

- Method: `GET`, `POST`, or `PATCH`
- URL: `https://127.0.0.1:8443/v1/...`
- Authentication: Header auth
- Header: `Authorization: Bearer <your-token>`
- TLS: trust the local certificate with the client's CA/certificate configuration; do not disable verification

Minimal example:

```bash
curl --silent --show-error --cacert /absolute/path/to/server-cert.pem \
  -H "Authorization: Bearer replace-me" \
  https://127.0.0.1:8443/v1/health
```

## Response Envelope

Success:

```json
{
  "ok": true,
  "data": {}
}
```

Error:

```json
{
  "ok": false,
  "error": {
    "code": "validation_error",
    "message": "..."
  }
}
```

Status code policy:

- `200` success
- `201` created
- `400` malformed request
- `401` missing or invalid bearer token
- `404` resource not found
- `409` semantic conflict
- `422` validation failure
- `429` too many authentication failures
- `500` internal unexpected failure
- `504` request timeout

## Security Headers

All success and error responses carry the same security baseline:

- `Strict-Transport-Security`
- `X-Content-Type-Options: nosniff`
- `Cache-Control: no-store`
- `Pragma: no-cache`

## ASVS Coverage Notes

The companion mapping in `docs/security/asvs-5.0.md` is a traceability matrix, not a
certification statement.

It is intended to show:

- which ASVS 5.0 controls are implemented directly in the HTTPS API
- which controls are only partially implemented
- which controls remain open or intentionally out of scope for this transport

It is not intended to guarantee full ASVS conformance for the repository or the deployment
environment around it.

## Request Logging

The server emits structured JSON logs aligned to the OpenTelemetry log data model:

- request lifecycle logs with method, route, status, duration, client IP, request id, and trace id
- security event logs for auth failures, invalid hosts, startup misconfiguration, and timeouts
- no `Authorization` values, WebDAV credentials, passphrases, or request bodies in normal logs

`traceparent` is accepted and propagated into the structured log fields when present.

## Endpoint Catalog

All endpoints live under `/v1`.

### Health and sync

- `GET /v1/health`
- `POST /v1/sync`

### Tasks

- `GET /v1/tasks`
- `GET /v1/tasks/search`
- `GET /v1/tasks/{task_id}`
- `POST /v1/tasks`
- `PATCH /v1/tasks/{task_id}`
- `POST /v1/tasks/{task_id}/complete`
- `POST /v1/tasks/{task_id}/drop`

Task filters:

- `inbox`
- `today`
- `flagged`
- `due`
- `project`
- `tag`
- `tag_id`
- `limit`

### Projects

- `GET /v1/projects`
- `GET /v1/projects/{project_id}`
- `POST /v1/projects`
- `PATCH /v1/projects/{project_id}`
- `POST /v1/projects/{project_id}/complete`

Project filters:

- `status`
- `tag`
- `tag_id`
- `limit`

### Project review

- `GET /v1/projects/review`
- `POST /v1/projects/{project_id}/review`

Review filters:

- `due_only`
- `limit`

### Folders

- `GET /v1/folders`
- `GET /v1/folders/tree`
- `GET /v1/folders/{folder_id}`
- `POST /v1/folders`
- `PATCH /v1/folders/{folder_id}`
- `POST /v1/folders/{folder_id}/drop`

### Tags

- `GET /v1/tags`
- `GET /v1/tags/{tag_id}`
- `POST /v1/tags`
- `PATCH /v1/tags/{tag_id}`
- `POST /v1/tags/{tag_id}/drop`

## Write Semantics

- reads may filter by human-readable names where the API supports that
- all write endpoints mutate by stable OmniFocus IDs in the path
- task and project tag assignment uses explicit `tag_ids`
- folder and tag moves validate parent existence, self-parenting, and cycles
- request bodies use Pydantic schema validation before reaching the service layer

## Operational Notes

- keep the API bound to localhost unless you have a reviewed network exposure plan
- prefer mounted PEM files over in-image cert material
- do not terminate TLS in plaintext in front of the app in this deployment shape
- do not expose the API externally without reviewing firewall rules, certificate posture, and host allowlists
- the default per-request timeout is 30 seconds
