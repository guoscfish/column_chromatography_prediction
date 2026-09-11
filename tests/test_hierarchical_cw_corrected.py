import numpy as np

from src.qgeognn_al.transfer.architecture_headroom import fit_hierarchical_center_width
from src.qgeognn_al.transfer.controlled_conditional_extension import center_width_inverse, center_width_transform
from src.qgeognn_al.transfer.hierarchical_cw_corrected import (
    CorrectedHierarchicalBasis,
    EndpointAlignedCorrectedFit,
    fit_corrected_joint_full128,
    fit_hierarchical_cw_endpoint_aligned_corrected,
    fit_hierarchical_cw_separate_lambda_legacy_objective,
)
from src.qgeognn_al.transfer.joint_hierarchical_latent import HierarchicalBasis, HierarchicalCWV2Fit


def _data():
    source = np.array([[2., 8.], [4., 14.], [8., 18.], [5., 17.], [10., 24.], [9., 27.]])
    ea = np.array([.1, .2, .3, .4, .5, .6])
    labels = np.array(["25g", "40g", "25g", "40g", "25g", "40g"])
    ratios = {"25g": 2.5, "40g": 4.0}
    truth = source * np.array([ratios[x] for x in labels])[:, None]
    return source, truth, ea, labels, ratios


def _identity_fit(source, ea, labels, ratios, center_scale, width_scale):
    fitted = CorrectedHierarchicalBasis.fit(source, ea)
    basis = CorrectedHierarchicalBasis(
        center_scale, width_scale, fitted.ea_mean, fitted.center_interaction_mean,
        fitted.center_interaction_scale, fitted.width_interaction_mean,
        fitted.width_interaction_scale,
    )
    size = 3 + 3 * len(ratios)
    center = np.zeros(size); width = np.zeros(size)
    center[0] = width[0] = 1.0
    return EndpointAlignedCorrectedFit(tuple(ratios), ratios, basis, center, width,
                                       np.ones(2), .1, .1)


def test_current_v2_unit_slope_is_not_mass_ratio_identity():
    source, _, ea, labels, ratios = _data()
    basis = HierarchicalBasis.fit(source, ea)
    size = 3 + 3 * len(ratios)
    center = np.zeros(size); width = np.zeros(size)
    center[0] = width[0] = 1.0
    current = HierarchicalCWV2Fit(tuple(ratios), ratios, basis, center, width, .1, .1, np.ones(2))
    expected = source * np.array([ratios[x] for x in labels])[:, None]
    actual = current.predict_raw(source, ea, labels)
    actual_center, actual_width = center_width_transform(actual)
    source_center, source_width = center_width_transform(source)
    ratio = np.array([ratios[x] for x in labels])
    np.testing.assert_allclose(actual_center[:, 0], ratio * source_center[:, 0] / basis.center_scale, atol=1e-12)
    np.testing.assert_allclose(actual_width[:, 0], ratio * source_width[:, 0] / basis.width_scale, atol=1e-12)
    assert not np.allclose(actual, expected)


def test_corrected_center_width_and_endpoint_identity_for_arbitrary_scales_and_ratios():
    source, _, ea, labels, ratios = _data()
    fit = _identity_fit(source, ea, labels, ratios, center_scale=7.3, width_scale=2.4)
    predicted = fit.predict_raw(source, ea, labels)
    expected = source * np.array([ratios[x] for x in labels])[:, None]
    source_c, source_w = center_width_transform(source)
    pred_c, pred_w = center_width_transform(predicted)
    ratio = np.array([ratios[x] for x in labels])
    np.testing.assert_allclose(pred_c[:, 0], ratio * source_c[:, 0], atol=1e-12)
    np.testing.assert_allclose(pred_w[:, 0], ratio * source_w[:, 0], atol=1e-12)
    np.testing.assert_allclose(predicted, expected, atol=1e-12)


def test_identity_has_zero_ea_and_column_deviation_contributions():
    source, _, ea, labels, ratios = _data()
    fit = _identity_fit(source, ea, labels, ratios, center_scale=3.7, width_scale=8.2)
    assert np.count_nonzero(fit.center_coefficients[1:]) == 0
    assert np.count_nonzero(fit.width_coefficients[1:]) == 0


def test_legacy_separate_lambda_exactly_nests_old_hier_when_equal():
    source, truth, ea, labels, ratios = _data()
    old = fit_hierarchical_center_width(source, truth, ea, labels, tuple(ratios),
                                        mass_ratios=ratios, base_penalty=.1, delta_penalty=1.0)
    new = fit_hierarchical_cw_separate_lambda_legacy_objective(
        source, truth, ea, labels, tuple(ratios), mass_ratios=ratios,
        base_penalty=.1, lambda_center_deviation=1.0, lambda_width_deviation=1.0,
    )
    np.testing.assert_allclose(new.predict_raw(source, ea, labels), old.predict_raw(source, ea, labels), atol=1e-12)


def test_corrected_joint_zero_latent_exactly_nests_corrected_hier():
    source, truth, ea, labels, ratios = _data()
    latent = np.arange(len(source) * 128, dtype=float).reshape(len(source), 128)
    plain = fit_hierarchical_cw_endpoint_aligned_corrected(
        source, truth, ea, labels, tuple(ratios), mass_ratios=ratios,
        base_penalty=.1, lambda_center_deviation=1., lambda_width_deviation=10.,
    )
    nested = fit_corrected_joint_full128(
        source, truth, ea, latent, labels, tuple(ratios), mass_ratios=ratios,
        base_penalty=.1, lambda_center_deviation=1., lambda_width_deviation=10.,
        lambda_center_latent=None, lambda_width_latent=None,
    )
    np.testing.assert_allclose(plain.predict_raw(source, ea, labels),
                               nested.predict_raw(source, ea, labels, latent), atol=0.0)
