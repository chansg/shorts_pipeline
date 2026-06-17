"""Landing-screen routing: the pure _route() helper that toggles the five
top-level containers, plus a headless build + initial-visibility check. No GPU."""
import app


def _visibles(updates):
    return [u.get("visible") for u in updates]


def test_route_shows_exactly_one_per_target():
    order = app._MODES  # ("landing", "shorts", "gaming", "fullauto", "settings")
    for i, target in enumerate(order):
        vis = _visibles(app._route(target))
        assert sum(bool(v) for v in vis) == 1, f"{target}: {vis}"
        assert vis[i] is True, f"{target} should show container #{i}: {vis}"


def test_route_back_to_origin():
    # Settings "Back" calls _route(prev_mode); each origin lands on its own view.
    # Order: [landing, shorts, gaming, fullauto, settings].
    assert _visibles(app._route("shorts")) == [False, True, False, False, False]
    assert _visibles(app._route("gaming")) == [False, False, True, False, False]
    assert _visibles(app._route("fullauto")) == [False, False, False, True, False]
    assert _visibles(app._route("landing")) == [True, False, False, False, False]


def test_build_app_constructs():
    demo = app.build_app()
    assert demo is not None


def test_landing_is_the_only_visible_container_on_load():
    # On load only the landing container is visible; the four mode containers are
    # hidden. Found unambiguously by their elem_id.
    ids = ("mode-landing", "mode-shorts", "mode-gaming", "mode-fullauto",
           "mode-settings")
    demo = app.build_app()
    by_id = {b.elem_id: b for b in demo.blocks.values()
             if getattr(b, "elem_id", None) in ids}
    assert set(by_id) == set(ids)
    assert by_id["mode-landing"].visible is True
    for hidden in ids[1:]:
        assert by_id[hidden].visible is False, f"{hidden} should be hidden on load"
