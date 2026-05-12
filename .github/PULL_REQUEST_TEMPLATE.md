<!-- Thanks for the PR. A few quick checks before submitting. -->

## What

<!-- One or two sentences on what this changes and why. -->

## How to verify

<!-- Commands / steps a reviewer can run locally. -->

```bash
pytest tests/test_xxx.py -v
```

## Checklist

- [ ] Tests added or updated for the change.
- [ ] `pytest` passes locally on Python 3.10+.
- [ ] If user-facing: added a line to `CHANGELOG.md` under `[Unreleased]`.
- [ ] If the change touches `agora_core/`: parity with the server is preserved.
- [ ] If the change affects the wire protocol: `PROTOCOL.md` is updated and a coordinated server-side change is referenced below.

## Related

<!-- Link related issues, server-side PR if any, etc. -->
