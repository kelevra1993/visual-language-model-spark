import pytest
import torch

from vision_training.models.sequence.sequence_comparator import SequenceComparator
from tests.fixtures.configuration_fixtures import get_sequence_comparison_model_configuration

# Fusions of the members and temporal modules covered by the comparison model tests
TESTED_FUSION_TYPES = ["concatenation", "difference", "cross_attention"]
TESTED_TEMPORAL_TYPES = ["mean_pooling", "attention_pooling", "gru", "lstm", "transformer", "temporal_convolution"]


@pytest.mark.parametrize(argnames="fusion_type", argvalues=TESTED_FUSION_TYPES)
@pytest.mark.parametrize(argnames="temporal_type", argvalues=TESTED_TEMPORAL_TYPES)
def test_sequence_comparator_outputs_and_padding_frames(temporal_type: str, fusion_type: str) -> None:
    """
    Verifies the outputs of the sequence comparator with every fusion and every kind of temporal module: [B, K] logits
    and softmax probabilities, the classification loss returned only with labels, and padding frames of both members
    that do not change the prediction of their pair whatever their pixels.

    Args:
        temporal_type (str): The type of the temporal module.
        fusion_type (str): The type of the fusion of the members.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type=temporal_type,
                                                                               fusion_type=fusion_type)
    model = SequenceComparator(configuration=configuration, device=device, dtype=dtype).eval()
    random_generator = torch.Generator().manual_seed(1)
    before_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator) * 255.0
    after_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator) * 255.0
    before_mask = torch.tensor(data=[[True, True, True, True], [True, True, False, False]])
    after_mask = torch.tensor(data=[[True, True, True, False], [True, False, False, False]])

    with torch.no_grad():
        outputs = model(before_images=before_images, after_images=after_images, before_mask=before_mask,
                        after_mask=after_mask)
        # White padding frames instead of noise ones, whose pooled features would clearly differ if they were used
        white_before_images, white_after_images = before_images.clone(), after_images.clone()
        white_before_images[~before_mask] = 255.0
        white_after_images[~after_mask] = 255.0
        white_outputs = model(before_images=white_before_images, after_images=white_after_images,
                              before_mask=before_mask, after_mask=after_mask)
        loss_outputs = model(before_images=before_images, after_images=after_images, before_mask=before_mask,
                             after_mask=after_mask, labels=torch.tensor(data=[0, 2]))

    assert outputs["class_logits"].shape == outputs["class_probabilities"].shape == (2, 3)
    assert torch.allclose(input=outputs["class_probabilities"].sum(dim=1), other=torch.ones(size=(2,)))
    assert "classification_loss" not in outputs and loss_outputs["classification_loss"].ndim == 0
    assert torch.allclose(input=outputs["class_logits"], other=white_outputs["class_logits"], atol=1e-5)


@pytest.mark.parametrize(argnames="fusion_type", argvalues=TESTED_FUSION_TYPES)
@pytest.mark.parametrize(argnames="temporal_type", argvalues=["gru", "transformer", "temporal_convolution",
                                                              "attention_pooling", "max_pooling"])
def test_mask_free_path_matches_the_masked_path(temporal_type: str, fusion_type: str) -> None:
    """
    Verifies that the deployment path without masks, where every member is encoded and summarised on its own with its
    real frames only, gives the predictions of the masked path of the training and the evaluation, where both members
    are padded to the same number of frames and processed together, the before and after members having different
    numbers of real frames. Computed in float64, so that only the order of the floating point operations differs.

    Args:
        temporal_type (str): The type of the temporal module.
        fusion_type (str): The type of the fusion of the members.
    """
    torch.manual_seed(seed=0)
    configuration, device, _ = get_sequence_comparison_model_configuration(temporal_type=temporal_type,
                                                                           fusion_type=fusion_type)
    model = SequenceComparator(configuration=configuration, device=device, dtype=torch.float64).eval()
    random_generator = torch.Generator().manual_seed(2)
    before_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator, dtype=torch.float64) * 255.0
    after_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator, dtype=torch.float64) * 255.0
    before_mask = torch.tensor(data=[[True, True, True, False]] * 2)
    after_mask = torch.tensor(data=[[True, True, False, False]] * 2)

    with torch.no_grad():
        masked_outputs = model(before_images=before_images, after_images=after_images, before_mask=before_mask,
                               after_mask=after_mask)
        mask_free_outputs = model(before_images=before_images[:, :3], after_images=after_images[:, :2])

    assert torch.allclose(input=masked_outputs["class_logits"], other=mask_free_outputs["class_logits"], atol=1e-9)


def test_batch_normalization_sees_both_members() -> None:
    """
    Verifies that the members of a pair are encoded together, so that in training mode the batch normalization
    statistics are computed over the frames of both members: the frame vectors of the before member then depend on
    the after frames, while in evaluation mode, with the running statistics, they do not.
    """
    configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type="gru",
                                                                               fusion_type="difference")
    model = SequenceComparator(configuration=configuration, device=device, dtype=dtype)
    random_generator = torch.Generator().manual_seed(3)
    before_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator) * 255.0
    first_after_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator) * 255.0
    sequence_mask = torch.ones(size=(2, 4), dtype=torch.bool)

    # Before frame vectors computed next to two different after members, in training then in evaluation mode
    for training_mode in [True, False]:
        model.train(mode=training_mode)
        with torch.no_grad():
            first_before_features, _ = model.encode_members(before_images=before_images,
                                                            after_images=first_after_images,
                                                            before_mask=sequence_mask, after_mask=sequence_mask)
            second_before_features, _ = model.encode_members(before_images=before_images,
                                                             after_images=first_after_images * 0.5,
                                                             before_mask=sequence_mask, after_mask=sequence_mask)
        members_are_independent = torch.allclose(input=first_before_features, other=second_before_features,
                                                 atol=1e-5)
        assert members_are_independent != training_mode


def test_padding_frames_never_reach_the_backbone() -> None:
    """
    Verifies that the padding frames of both members never reach the backbone: with the batch normalization in
    training mode, padding frames made of NaN would spread NaN to the loss and the gradients through the batch
    statistics if they did.
    """
    configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type="transformer",
                                                                               fusion_type="cross_attention")
    model = SequenceComparator(configuration=configuration, device=device, dtype=dtype).train()
    random_generator = torch.Generator().manual_seed(4)
    before_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator) * 255.0
    after_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator) * 255.0
    before_mask = torch.tensor(data=[[True, True, True, False], [True, False, False, False]])
    after_mask = torch.tensor(data=[[True, True, False, False], [True, True, True, True]])
    before_images[~before_mask] = float("nan")
    after_images[~after_mask] = float("nan")

    outputs = model(before_images=before_images, after_images=after_images, before_mask=before_mask,
                    after_mask=after_mask, labels=torch.tensor(data=[1, 0]))
    outputs["classification_loss"].backward()

    assert torch.isfinite(input=outputs["classification_loss"])
    assert all(torch.isfinite(input=parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


@pytest.mark.parametrize(argnames=("fusion_type", "members_are_mixed"), argvalues=[("difference", False),
                                                                                 ("cross_attention", True)])
def test_only_the_cross_attention_mixes_the_members(fusion_type: str, members_are_mixed: bool) -> None:
    """
    Verifies that the vector summarising the before member depends on the after frames only with the cross attention
    fusion, the other fusions summarising every member on its own before combining them (evaluation mode, so that the
    batch normalization does not relate the members).

    Args:
        fusion_type (str): The type of the fusion of the members.
        members_are_mixed (bool): Whether the before vectors are expected to depend on the after frames.
    """
    torch.manual_seed(seed=0)
    configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type="mean_pooling",
                                                                               fusion_type=fusion_type)
    model = SequenceComparator(configuration=configuration, device=device, dtype=dtype).eval()
    random_generator = torch.Generator().manual_seed(5)
    before_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator) * 255.0
    first_after_images = torch.rand(size=(2, 4, 3, 64, 64), generator=random_generator) * 255.0
    sequence_mask = torch.ones(size=(2, 4), dtype=torch.bool)

    # Vector of the before member next to two different after members
    before_vectors = []
    with torch.no_grad():
        for after_images in [first_after_images, first_after_images.flip(dims=[3])]:
            before_features, after_features = model.encode_members(before_images=before_images,
                                                                   after_images=after_images,
                                                                   before_mask=sequence_mask, after_mask=sequence_mask)
            before_features, after_features = model.member_fusion.attend_across_members(
                before_features=before_features, after_features=after_features, before_mask=sequence_mask,
                after_mask=sequence_mask)
            member_vectors = model.summarise_members(before_features=before_features, after_features=after_features,
                                                     before_mask=sequence_mask, after_mask=sequence_mask)
            before_vectors.append(member_vectors[0])

    assert torch.allclose(input=before_vectors[0], other=before_vectors[1], atol=1e-6) != members_are_mixed


def test_a_single_mask_is_rejected() -> None:
    """
    Verifies that the masks of the two members must be given together: the masked path processes both members as one
    batch, the mask-free path processes them separately, and mixing them would be ambiguous.
    """
    configuration, device, dtype = get_sequence_comparison_model_configuration(temporal_type="gru",
                                                                               fusion_type="concatenation")
    model = SequenceComparator(configuration=configuration, device=device, dtype=dtype).eval()
    frames = torch.rand(size=(1, 4, 3, 64, 64)) * 255.0

    with pytest.raises(expected_exception=ValueError, match="given together"):
        model(before_images=frames, after_images=frames, before_mask=torch.ones(size=(1, 4), dtype=torch.bool))
