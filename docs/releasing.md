# Publishing releases

The repository produces two independently versioned PyPI distributions:

| Distribution | Source directory | Release tag |
| --- | --- | --- |
| symmetry-learn-default-model | model_packages/symmetry_learn_default_model | model-v1.0.0 |
| symmetry-learn | repository root | v0.1.0 |

## One-time account setup

Sign in to PyPI with a verified email and two-factor authentication. Under
<https://pypi.org/manage/account/publishing/>, add a pending GitHub publisher
for **each** distribution, using these fields:

- PyPI project name: `symmetry-learn-default-model` or `symmetry-learn`
- Owner: `jiadongdan`
- Repository: `symmetry-learn`
- Workflow filename: `publish.yml`
- Environment: `pypi`

Create the `pypi` GitHub environment in the repository settings and configure
a required reviewer for releases. No PyPI token or repository secret is needed.
Pending publishers do not reserve a package name until the first upload.

## Release checks

Use Python 3.11+ for release tooling. This does not change the library's
Python >=3.9 requirement. Fetch Git LFS objects before building; the checkpoint
must be the real binary, not a pointer file.

```bash
git lfs pull
python -m pip install build twine
python -m build --outdir dist/model model_packages/symmetry_learn_default_model
python -m build --outdir dist/library .
python -m twine check --strict dist/model/* dist/library/*
python scripts/check_release.py
```

Use a fresh output directory for each release; the checker rejects stale or
additional archives. The workflow repeats these checks, installs both wheels
in a clean environment outside the checkout, and strictly loads the default
checkpoint. A manual workflow run validates and retains artifacts without
publishing anything. Pull requests affecting packaging also validate builds.

## First publication

Commit and push the prepared release files. Run **Build and publish distributions**
manually on the intended commit and review its results. Then publish the model:

```bash
git tag model-v1.0.0
git push origin model-v1.0.0
```

Approve the `pypi` environment job. Wait for the model wheel to appear on PyPI
before publishing the library:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Approve its `pypi` environment job. Tag versions must match package metadata.
The library publishing job verifies that its exact model dependency has an
available, non-yanked wheel on PyPI. The build job only grants read access;
short-lived upload credentials are available only to the publishing jobs.

Finally verify `pip install symmetry-learn` in a new environment and run
`symmetry-learn-provider --probe` with a valid probe job, or call
`probe_model(options={"device": "cpu"})` after importing it from
`symmlearn.provider.api` in Python.

## Later releases

For code-only releases, update the root `pyproject.toml`, `symmlearn.__version__`,
and version assertions in tests, then push a new `v...` tag. Keep model version
1.0.0 unchanged while the checkpoint is unchanged; no model upload is needed.

For new model weights, update the model package version, manifest, registered
weight metadata (including size and SHA-256), and the library's pinned dependency.
Publish the new model version first and then a new library version. Already
published files cannot be replaced: use a new version for corrected artifacts.

Official reference: <https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/>.
