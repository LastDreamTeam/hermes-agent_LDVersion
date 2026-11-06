"""Runtime contract: multiplexed subscriptions cannot share the host's grant."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from agent.secret_scope import set_multiplex_active, UnscopedSecretError
from gateway.run import _profile_runtime_scope
from hermes_cli.auth_nous import (
    _merge_shared_nous_oauth_state,
    _nous_shared_store_path,
    _write_shared_nous_state,
)


def test_concurrent_profile_refresh_uses_its_own_shared_store(tmp_path, monkeypatch):
    host_store = tmp_path / 'host-subscription'
    monkeypatch.setenv('HERMES_SHARED_AUTH_DIR', str(host_store))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'root'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    barrier = threading.Barrier(2)
    profiles = []
    for name in ('subscription-a', 'subscription-b'):
        home = tmp_path / 'profiles' / name
        home.mkdir(parents=True)
        shared = tmp_path / 'shared' / name
        (home / '.env').write_text(f'HERMES_SHARED_AUTH_DIR={shared}\n')
        (home / 'config.yaml').write_text('{}\n')
        profiles.append((name, home, shared))
    set_multiplex_active(True)
    try:
        def refresh(item):
            name, home, shared = item
            with _profile_runtime_scope(home, hydrate_secrets=False):
                assert _nous_shared_store_path() == shared / 'nous_auth.json'
                _write_shared_nous_state({'access_token': name, 'refresh_token': name + '-refresh'})
                barrier.wait(timeout=15)
                local = {'access_token': 'old', 'refresh_token': 'old'}
                assert _merge_shared_nous_oauth_state(local)
                assert local['access_token'] == name
                assert local['refresh_token'] == name + '-refresh'
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(refresh, profiles))
        assert not host_store.exists()
    finally:
        set_multiplex_active(False)


def test_unscoped_multiplex_rejects_host_store_but_standalone_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_SHARED_AUTH_DIR', str(tmp_path / 'standalone'))
    set_multiplex_active(True)
    try:
        with pytest.raises(UnscopedSecretError):
            _nous_shared_store_path()
    finally:
        set_multiplex_active(False)
    assert _nous_shared_store_path() == tmp_path / 'standalone' / 'nous_auth.json'
