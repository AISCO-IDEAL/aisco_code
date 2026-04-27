import pytest
import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.weights import weights_alm


def test_weights_alm_relative_import_and_callable():
    assert callable(weights_alm)


def test_weights_alm_no_grad_graph_isolation_and_in_place_updates():
    lamdas = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    mus = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    nus = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    alphas = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)
    constrained_losses = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64, requires_grad=True)

    lam_ptr = lamdas.data_ptr()
    mu_ptr = mus.data_ptr()
    nu_ptr = nus.data_ptr()

    out_lamdas, out_mus, out_nus = weights_alm(
        lamdas=lamdas,
        mus=mus,
        nus=nus,
        alphas=alphas,
        gamma=1.0,
        gamma_aggressive=1.1,
        constrained_losses=constrained_losses,
        thresholds=None,
        use_alm_aggressive=False,
    )

    # In-place memory path integrity
    assert out_lamdas.data_ptr() == lam_ptr
    assert out_mus.data_ptr() == mu_ptr
    assert out_nus.data_ptr() == nu_ptr

    # No computational graph nodes created by internal updates
    assert out_lamdas.grad_fn is None
    assert out_mus.grad_fn is None
    assert out_nus.grad_fn is None

    # Backward on lambda-only scalar should not propagate to constrained_losses
    scalar = out_lamdas.sum()
    scalar.backward()
    assert constrained_losses.grad is None
    assert lamdas.grad is not None


@pytest.mark.parametrize(
    "thresholds",
    [
        None,
        torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64),
    ],
)
def test_weights_alm_branches_keep_graph_disconnected(thresholds):
    lamdas = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    mus = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    nus = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    alphas = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    constrained_losses = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64, requires_grad=True)

    out_lamdas, out_mus, out_nus = weights_alm(
        lamdas=lamdas,
        mus=mus,
        nus=nus,
        alphas=alphas,
        gamma=1.0,
        gamma_aggressive=1.1,
        constrained_losses=constrained_losses,
        thresholds=None,
        use_alm_aggressive=False,
    )

    assert out_lamdas.shape == lamdas.shape
    assert out_mus.shape == mus.shape
    assert out_nus.shape == nus.shape
    assert torch.isfinite(out_lamdas).all()
    assert torch.isfinite(out_mus).all()
    assert torch.isfinite(out_nus).all()

    out_lamdas.sum().backward()
    assert constrained_losses.grad is None


def test_weights_alm_empty_tensors_edge_case():
    lamdas = torch.tensor([], dtype=torch.float64, requires_grad=True)
    mus = torch.tensor([], dtype=torch.float64)
    nus = torch.tensor([], dtype=torch.float64)
    alphas = torch.tensor([], dtype=torch.float64)
    constrained_losses = torch.tensor([], dtype=torch.float64, requires_grad=True)

    out_lamdas, out_mus, out_nus = weights_alm(
        lamdas=lamdas,
        mus=mus,
        nus=nus,
        alphas=alphas,
        gamma=1.0,
        gamma_aggressive=1.1,
        constrained_losses=constrained_losses,
        thresholds=None,
        use_alm_aggressive=False,
    )

    assert out_lamdas.numel() == 0
    assert out_mus.numel() == 0
    assert out_nus.numel() == 0

    # Returned tensors are the same objects (in-place semantics)
    assert out_lamdas is lamdas
    assert out_mus is mus
    assert out_nus is nus


def test_weights_alm_shape_consistency_and_broadcasting_protection():
    """
    Überprüft, ob die Funktion auch dann korrekte 1D oder konsistente Shapes zurückgibt,
    wenn Eingaben als Spaltenvektoren [3, 1] und Verluste als [3] übergeben werden.
    Dies verhindert die 'Matrix-Explosion' auf [3, 3].
    """
    # Simuliere TotalLoss Initialisierung: Spaltenvektoren [3, 1]
    lamdas = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float64)
    mus = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float64)
    nus = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float64)
    alphas = torch.tensor([[0.9], [0.8], [0.7]], dtype=torch.float64)

    # Simuliere c_for_alm: 1D Vektor [3]
    constrained_losses = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)

    out_lamdas, out_mus, out_nus = weights_alm(
        lamdas=lamdas,
        mus=mus,
        nus=nus,
        alphas=alphas,
        gamma=1.0,
        gamma_aggressive=1.1,
        constrained_losses=constrained_losses,
        thresholds=None,
        use_alm_aggressive=False,  # Testet RMS Zweig
    )

    # WICHTIGSTER CHECK: Die Shape darf nicht auf [3, 3] explodiert sein!
    # Sie sollte entweder [3, 1] bleiben (wenn in-place view genutzt wurde)
    # oder [3] sein (wenn komplett flachgedrückt wurde).
    # Hauptsache, die Anzahl der Elemente ist weiterhin 3.
    assert out_lamdas.numel() == 3, f"Broadcasting Fehler: Lamdas hat {out_lamdas.numel()} Elemente statt 3!"
    assert out_mus.numel() == 3
    assert out_nus.numel() == 3

    # Sicherstellen, dass keine Matrix entstanden ist (Shape check)
    assert len(out_lamdas.shape) <= 2
    if len(out_lamdas.shape) == 2:
        assert out_lamdas.shape[1] == 1, "Zweite Dimension darf nicht expandiert sein (Broadcasting Error)!"
