import pytest
import torch

from vision_training.models.sequence.member_fusion import MemberFusion


@pytest.mark.parametrize(argnames=("fusion_type", "expected_output_features"), argvalues=[
    ("concatenation", 6), ("difference", 9), ("cross_attention", 9)])
def test_member_vectors_are_combined(fusion_type: str, expected_output_features: int) -> None:
    """
    Verifies the combination of the member vectors given to the classification head: their concatenation [a, b], or
    [a, b, b - a] with the difference making the change between the members explicit, and the matching number of
    output features.

    Args:
        fusion_type (str): The type of the fusion.
        expected_output_features (int): The expected number of features of the combined vectors, for 3 features per
            member.
    """
    fusion = MemberFusion(fusion_type=fusion_type, frame_features=8, member_features=3, number_heads=2,
                          dropout_rate=0.0, device=torch.device(device="cpu"), dtype=torch.float32)
    before_vectors = torch.tensor(data=[[1.0, 2.0, 3.0]])
    after_vectors = torch.tensor(data=[[4.0, 6.0, 8.0]])

    combined_vectors = fusion.combine_member_vectors(before_vectors=before_vectors, after_vectors=after_vectors)

    assert fusion.output_features == expected_output_features == combined_vectors.shape[1]
    expected_values = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0] + ([] if fusion_type == "concatenation" else [3.0, 4.0, 5.0])
    assert combined_vectors[0].tolist() == expected_values


@pytest.mark.parametrize(argnames="fusion_type", argvalues=["concatenation", "difference"])
def test_fusions_without_attention_leave_the_frames_unchanged(fusion_type: str) -> None:
    """
    Verifies that the fusions without cross attention give the frame vectors of the members to the temporal module
    unchanged, and hold no weight.

    Args:
        fusion_type (str): The type of the fusion.
    """
    fusion = MemberFusion(fusion_type=fusion_type, frame_features=8, member_features=3, number_heads=2,
                          dropout_rate=0.0, device=torch.device(device="cpu"), dtype=torch.float32)
    before_features, after_features = torch.rand(size=(2, 4, 8)), torch.rand(size=(2, 3, 8))

    attended_before, attended_after = fusion.attend_across_members(before_features=before_features,
                                                                   after_features=after_features, before_mask=None,
                                                                   after_mask=None)

    assert attended_before is before_features and attended_after is after_features
    assert len(list(fusion.parameters())) == 0


def test_cross_attention_ignores_padding_frames() -> None:
    """
    Verifies that the cross attention updates the real frames of each member with the frames of the other member,
    ignores the padding frames of the other member whatever their vectors, and gives zero vectors at the padding frames
    of the updated member, so that the temporal module still ignores them.
    """
    torch.manual_seed(seed=0)
    fusion = MemberFusion(fusion_type="cross_attention", frame_features=8, member_features=3, number_heads=2,
                          dropout_rate=0.0, device=torch.device(device="cpu"), dtype=torch.float32).eval()
    before_features, after_features = torch.rand(size=(2, 4, 8)), torch.rand(size=(2, 4, 8))
    before_mask = torch.tensor(data=[[True, True, True, False], [True, True, False, False]])
    after_mask = torch.tensor(data=[[True, True, False, False], [True, True, True, True]])

    with torch.no_grad():
        attended_before, attended_after = fusion.attend_across_members(before_features=before_features,
                                                                       after_features=after_features,
                                                                       before_mask=before_mask, after_mask=after_mask)
        # Padding frames of both members replaced by huge vectors, which would change the attention if attended to
        noisy_before, noisy_after = before_features.clone(), after_features.clone()
        noisy_before[~before_mask] = 1000.0
        noisy_after[~after_mask] = 1000.0
        noisy_attended_before, noisy_attended_after = fusion.attend_across_members(
            before_features=noisy_before, after_features=noisy_after, before_mask=before_mask, after_mask=after_mask)

    assert torch.allclose(input=attended_before[before_mask], other=noisy_attended_before[before_mask], atol=1e-5)
    assert torch.allclose(input=attended_after[after_mask], other=noisy_attended_after[after_mask], atol=1e-5)
    assert torch.equal(input=attended_before[~before_mask], other=torch.zeros(size=(3, 8)))
    assert torch.equal(input=attended_after[~after_mask], other=torch.zeros(size=(2, 8)))
    assert not torch.allclose(input=attended_before[before_mask], other=before_features[before_mask])
