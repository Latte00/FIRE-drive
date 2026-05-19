from dataclasses import dataclass
from re import T
from typing import Tuple, List, Optional
from enum import Enum

import numpy as np
try:
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
except ImportError:
    class SemanticMapLayer(Enum):
        LANE = "LANE"
        INTERSECTION = "INTERSECTION"
        WALKWAYS = "WALKWAYS"
        LANE_CONNECTOR = "LANE_CONNECTOR"
        BASELINE_PATHS = "BASELINE_PATHS"

    class TrackedObjectType(Enum):
        CZONE_SIGN = "CZONE_SIGN"
        BARRIER = "BARRIER"
        TRAFFIC_CONE = "TRAFFIC_CONE"
        GENERIC_OBJECT = "GENERIC_OBJECT"
        VEHICLE = "VEHICLE"
        PEDESTRIAN = "PEDESTRIAN"
        BICYCLE = "BICYCLE"
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from timm.layers import F


@dataclass
class TransfuserConfig:
    """Global TransFuser config."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    ego_fut_mode: int = 20  # ancher妯℃€佹暟閲?
    bev_semantic_pretrained_path: Optional[str] = None #"/home/xqf/DiffusionDrive-main/exp/training_diffusiondrive_agent/best_bev/lightning_logs/version_0/checkpoints/best_bev.ckpt"  # backbone鐨刢kp璺緞
    freeze_bev_semantic_backbone: bool = False  # 鏄惁鍐荤粨鏁翠釜backbone
    freeze_bev_semantic_head: bool = False  # 鏄惁鍐荤粨semantic head璇箟澶达紙閮ㄥ垎backbone锛?

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    bkb_path: str = "assets/resnet34.a1_in1k/pytorch_model.bin"
    image_backbone_mode: str = "transfuser"
    image_vit_ckpt: str = ""
    vit_encoder: str = "vitl"
    vit_image_channels: int = 1024
    vit_late_fuse_weight: float = 1.0
    vit_late_fuse_use_gate: bool = True
    plan_anchor_path: str = "assets/kmeans_navsim_traj_20.npy"

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0
    force_train_forward: bool = False  # use train forward path even in eval

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 0.0
    trajectory_reg_weight: float = 8.0
    trajectory_selected_weight: float = 0.0  # loss on selected (argmax) trajectory vs GT
    trajectory_cls_use_future_bev: bool = False  # use future BEV to supervise mode scores锛坈ls锛?
    trajectory_cls_future_temp: float = 1.0  # temperature for future BEV mode supervision
    trajectory_cls_future_progress_weight: float = 1.0  # weight for forward progress score
    trajectory_cls_future_feasible_weight: float = 1.0  # weight for feasible ratio score
    trajectory_cls_current_bev_steps: int = 3  # use current BEV for first N steps
    trajectory_cls_gt_soft_weight: float = 0.0  # mix weight for GT-distance soft targets
    trajectory_cls_gt_soft_temp: float = 3.0  # temperature for GT-distance soft targets
    trajectory_progress_weight: float = 0.0  # aux loss on forward pred progress from future BEV锛坧red锛?
    trajectory_progress_branch_matching: bool = True  # match modes to lane branches for progress target
    trajectory_progress_mode_weighted: bool = True  # weight progress loss by mode probabilities
    trajectory_progress_max_branches: int = 5  # cap number of lane branches used for matching
    diff_loss_weight: float = 20.0
    diff_input_decorrelation_weight: float = 0.0  # 鍘荤浉鍏砫ecorrelate diffusion input features (traj_feature)
    diff_output_decorrelation_weight: float = 0.0  # decorrelate decoder features before task head
    cross_bev_decorrelation_weight: float = 0.0  # 鍘荤浉鍏砫ecorrelate cross_bev_feature channels
    cross_bev_decorrelation_stride: int = 4  # 涓嬮噰鏍锋闀匡紝鍑忓皯璁＄畻閲弒patial downsample stride for decorrelation
    denoise_use_image_tokens: bool = False  # 鍘诲櫔瑙ｇ爜鍣ㄦ敞鍏ュ浘鍍忕壒寰乼oken
    denoise_use_lidar_tokens: bool = False  # 鍘诲櫔瑙ｇ爜鍣ㄦ敞鍏ラ浄杈剧壒寰乼oken
    denoise_use_history_tokens: bool = True  # 鍘诲櫔瑙ｇ爜鍣ㄦ敞鍏ヨ嚜杞﹀巻鍙茶建杩箃oken
    denoise_use_ego_query: bool = True  # 鍘诲櫔瑙ｇ爜鍣ㄦ敞鍏go query
    denoise_use_time_embed: bool = False  # 鍘诲櫔瑙ｇ爜鍣ㄤ娇鐢ㄦ墿鏁ｆ椂闂存宓屽叆/璋冨埗
    denoise_norm_bev: bool = False  # cross_bev_attention鍚庢槸鍚﹀仛LayerNorm
    diff_decoder_layers: int = 3  # diffusion decoder鍫嗗彔灞傛暟锛岄粯璁?
    trajectory_decoder_type: str = "diffusion"  # "diffusion" for the legacy decoder, "proformer" for ProFormerDrive
    proformer_ref_num: int = 4  # number of proposal refinement rounds in ProFormerDrive
    proformer_use_risk_feedback: bool = True  # inject current risk features back into proposal refinement
    proformer_detach_proposal_ref: bool = True  # detach proposal coordinates between refinement rounds
    proformer_pure_visual_mode: bool = False  # ignore lidar input/features for ProFormerDrive
    proformer_use_global_bev: bool = True  # attend to the full BEV token map each round
    proformer_disable_actor_query: bool = True  # do not use actor queries in ProFormerDrive rounds
    proformer_risk_feedback_prev_only: bool = True  # inject only previous-round risk context
    proformer_risk_feedback_detach: bool = True  # detach previous-round risk context before next round
    proformer_status_fuse_pre_loop: bool = True  # fuse status encoding once before the ProFormer loop
    proformer_delta_clip_xy: float = 0.0  # 0 disables residual clipping
    proformer_delta_clip_heading: float = 0.0  # 0 disables heading residual clipping
    proformer_risk_aux_enable: bool = False  # supervise each ProFormer round with fast NC/DAC risk targets
    proformer_risk_aux_weight: float = 0.0  # global weight for ProFormer fast-risk auxiliary loss
    proformer_risk_aux_nc_weight: float = 1.0  # relative weight for no-collision risk target
    proformer_risk_aux_dac_weight: float = 1.0  # relative weight for drivable-area risk target
    proformer_risk_collision_margin_m: float = 0.0  # extra margin for fast NC supervision
    proformer_risk_collision_mode: str = "any"  # "any" or "front" for NAVSIM fast NC supervision
    # Trajectory token initialization:
    # token = anchor_token + learned_mode_token + ego_token.
    drivor_token_init_enable: bool = False  # 鏄惁鍚敤娉ㄥ唽token鐨勬€诲紑鍏?
    drivor_token_init_use_anchor: bool = False
    drivor_token_init_disable_anchor: bool = True  # hard switch: ignore anchor token in token init
    drivor_token_init_use_learned_mode: bool = True
    drivor_token_init_use_ego: bool = True
    drivor_token_anchor_weight: float = 1.0
    drivor_token_mode_weight: float = 1.0
    drivor_token_ego_weight: float = 1.0
    anchor_contrastive_enable: bool = False  # supervise anchor modes by GT-nearest positive matching
    anchor_contrastive_weight: float = 1.0  # weight of anchor contrastive loss
    anchor_contrastive_temperature: float = 0.07  # temperature for anchor/GT similarity logits
    
    kinematic_residual_enable: bool = True  # 鍚敤涓嬩竴姝ョ墿鐞嗘畫宸娴嬫ā鍧?
    kinematic_residual_weight: float = 1.0  # 涓嬩竴姝ユ畫宸洃鐫ｆ崯澶辨潈閲?
    kinematic_residual_hidden_dim: int = 128  # 娈嬪樊棰勬祴MLP闅愯棌缁村害
    kinematic_residual_use_history: bool = True  # 娈嬪樊棰勬祴鏄惁鎷兼帴鍘嗗彶鐘舵€?
    kinematic_residual_as_condition: bool = True  # 灏嗕笅涓€姝ラ娴嬫敞鍏tatus_encoding
    output_bev_feature: bool = False  # 杈撳嚭cross_bev_feature鐢ㄤ簬澶栭儴缂撳瓨/鍒嗘瀽
    
    eval_use_predicted_bev_masks: bool = True  # 楠岃瘉闃舵浣跨敤棰勬祴BEV鎻愬彇鍙鍩?
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    # BEV杈呭姪鎹熷け鏉冮噸
    bev_centerline_aux_weight: float = 0.5  # centerline BCE+Dice auxiliary loss weight
    bev_drivable_aux_weight: float = 0.5  # drivable BCE+Dice auxiliary loss weight
    bev_aux_bce_weight: float = 1.0  # BCE weight for BEV aux losses
    bev_aux_dice_weight: float = 1.0  # Dice weight for BEV aux losses
    loss_nonfinite_guard_enable: bool = False  # skip optimizer update when total loss is non-finite
    bev_loss_logit_guard_enable: bool = False  # clamp BEV logits before CE to avoid AMP overflow
    
    use_ema: bool = False
    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
    bev_road_label: int = 1
    bev_centerline_label: int = 3
    bev_static_label: int = 4  # static obstacle class id in BEV semantic map
    bev_vehicle_label: int = 5  # vehicle class id in BEV semantic map
    bev_pedestrian_label: int = 6  # pedestrian class id in BEV semantic map
    # 鎻愬彇鍙杞﹂亾绾垮尯鍩?
    extract_feasible_lane: bool = True  # extract feasible lane and area from BEV semantic map
    feasible_lane_seed_rows: int = 12
    feasible_lane_seed_cols: int = 16
    feasible_lane_max_iters: int = 0
    # 鎻愬彇杞﹂亾绾跨敓鎴愬€欓€塯t锛岀洿鎺ヤ娇鐢ㄤ細鏇存柊缂撳瓨鏁版嵁
    generate_trajectory_candidates: bool = False  # build candidate GT trajectories
    trajectory_candidates_mode: str = "centerline"  # "centerline" or "semantic"(鍩轰簬杞﹂亾绾?寮虹害鏉?鎴栬涔?
    trajectory_candidates_include_gt: bool = True  # include GT as candidate 0
    trajectory_candidates_max_count: int = 4
    trajectory_candidates_lane_change_offset_m: float = 3.5  # semantic lane-change offset
    trajectory_candidates_lane_change_use_gt_offset: bool = True  # use GT lateral shift as offset estimate
    trajectory_candidates_lane_change_min_m: float = 3.0  # clamp GT offset (min)
    trajectory_candidates_lane_change_max_m: float = 4.5  # clamp GT offset (max)
    trajectory_candidates_yaw_rate_window: int = 3  # steps to estimate yaw rate for keep-lane
    trajectory_candidates_search_radius: float = 60.0
    trajectory_candidates_max_heading_error_deg: float = 60.0
    trajectory_candidates_max_lateral_error: float = 5.0
    trajectory_candidates_min_length_ratio: float = 0.6
    trajectory_candidates_tangent_eps: float = 1.0
    trajectory_candidates_start_blend_steps: int = 5  # blend ego -> centerline at the start
    trajectory_candidates_filter_by_feasible: bool = True  # filter candidates by feasible area
    trajectory_candidates_feasible_ratio: float = 0.6  # min in-mask ratio to keep candidate
    trajectory_candidates_feasible_step_m: float = 1.0  # sampling step along trajectory
    trajectory_candidates_filter_by_dynamic: bool = True  # filter candidates by dynamic obstacles
    trajectory_candidates_dynamic_filter_steps: int = 3  # check first N steps for dynamic overlap

    trajectory_candidate_weight: float = 1.0  # soft supervision weight for candidate GTs
    trajectory_candidate_cls_weight: float = 0.0  # extra KL weight to bias cls toward high-quality candidates
    trajectory_candidate_cls_temp: float = 1.0  # temperature for candidate cls soft targets
    trajectory_candidate_exclude_gt_mode: bool = True  # exclude GT-matched mode from candidate supervision
    trajectory_candidate_softmin_temp: float = 1.0  # temperature for candidate softmin
    trajectory_candidate_mode_temp: float = 1.0  # temperature for mode softmin
    trajectory_candidate_max_yaw_flips: int = 0  # 0 disables; drop candidates with more flips 闄愬埗杞集娆℃暟
    trajectory_candidate_max_dyaw: float = 1.5  # 0 disables; max per-step yaw change (rad) 鏈€灏忚搴﹀彉鍖?
    trajectory_candidate_max_lat_flips: int = 1  # 0 disables; flips of lateral delta 闄愬埗妯悜鍋忕Щ娆℃暟
    trajectory_candidate_lat_eps: float = 0.5  # lateral flip epsilon (meters) 鍙帴鍙楃殑妯悜鍋忕Щ闃堝€?

    pdm_score_weight: float = 1.0  # weight for PDM score supervision
    pdm_score_use_head: bool = True  # enable learned PDM scorer head on proposals
    pdm_score_use_for_selection: bool = True  # use PDM scorer to select mode in inference锛堟帹鐞嗭級
    pdm_score_select_topk: int = 0  # 鎺ㄧ悊 : select top-k by poses_cls then rerank by PDM score (0=off)
    pdm_score_topk: int = 0  # 璁粌锛歴core top-k modes (0 = all)
    pdm_score_temp: float = 1.0  # softmax temperature for PDM score targets
    pdm_metric_cache_path: Optional[str] = None  # metric cache path for PDM supervision
    pdm_metric_cache_path_monitor: Optional[str] = None  # optional metric cache path override used by monitor dataloader validation
    pdm_score_train_only: bool = False  # only apply PDM supervision during training
    pdm_score_use_components: bool = True  # predict 6 PDM component scores per mode
    pdm_score_component_weights: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )  # no_collision, drivable, progress, ttc, comfort, driving_direction
    pdm_score_use_decoder: bool = True  # use trajectory-query decoder scorer
    pdm_score_decoder_layers: int = 2  # decoder layers for scorer
    pdm_score_decoder_heads: int = 4  # attention heads for scorer
    pdm_score_decoder_dropout: float = 0.1  # dropout for scorer decoder
    pdm_score_traj_dim: int = 3  # scorer uses (x, y, heading) when available
    pdm_score_bev_pool_hw: Tuple[int, int] = (8, 16)  # BEV token grid when no scene tokens
    pdm_score_use_cached_poses: bool = False  # use cached poses_reg for scorer head input
    pdm_score_cached_pose_align_weight: float = 0.0  # supervise online poses vs cached poses
    pdm_score_use_cached_targets_in_train: bool = False  # use cached PDM targets during training
    pdm_score_use_cached_targets_in_val: bool = False  # use cached PDM targets during validation
    pdm_score_use_image_tokens: bool = False  # use image tokens in scorer only
    pdm_score_use_lidar_tokens: bool = False  # use lidar tokens in scorer only
    pdm_score_use_agent_tokens: bool = True  # use agent query tokens in scorer only
    pdm_score_head_only: bool = False  # train only the PDM scorer head (freeze all other params)
    pdm_score_use_ray: bool = False  # use ray to parallelize PDM score computation
    pdm_score_ray_threads: int = 32  # number of threads for ray
    pdm_score_cache_lru_size: int = 128  # size of LRU cache for PDM score      
    pdm_score_use_offline_targets: bool = False  # use cached PDM scores in targets
    pdm_score_offline_assign_temp: float = 1.0  # temp for matching preds to candidates
    pdm_score_offline_assign_max_dist_m: float = 0.0  # 0 disables; drop matches beyond this mean L2 (meters)
    pdm_score_component_kl_weight: float = 0.0  # weight for KL loss on components
    pdm_score_component_bce_weight: float = 0.0  # extra BCE loss on component logits
    # 瀵箂corer澧炲姞 pairwise 鎺掑悕鎹熷け,瀵筽dm鐨勬暣浣撴帓搴?
    pdm_score_pairwise_enable: bool = False  # add pairwise ranking loss on mode scores
    pdm_score_pairwise_weight: float = 1.0  # weight for pairwise ranking loss
    pdm_score_pairwise_min_target_gap: float = 0.02  # ignore mode pairs with tiny target gap
    pdm_score_pairwise_margin: float = 0.0  # >0 uses hinge margin ranking, else logistic ranking
    pdm_score_pairwise_max_pairs: int = 0  # max pairs per sample (0 means all)
    pdm_score_component_pos_weight: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )  # per-component positive weight for BCE
    pdm_score_component_neg_weight: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )  # per-component negative weight for BCE
    pdm_score_component_pos_thresh: float = 0.5  # threshold for hard negative mining
    pdm_score_component_hard_neg_topk: int = 3  # top-k hard negatives per component (0=off)
    pdm_score_use_logsigmoid_aggregate: bool = True  # aggregate PDM components using DrivoR-style log-sigmoid product
    pdm_score_progress_use_reference_baseline: bool = False  # normalize progress component against official PDM baseline trajectory during online supervision
    b2d_pdm_score_enable: bool = False  # compute B2D pseudo-PDM labels online from cache targets instead of NAVSIM metric cache
    b2d_pdm_bev_pixel_size: float = 0.25
    b2d_pdm_bev_front_m: float = 32.0
    b2d_pdm_bev_left_m: float = 32.0
    b2d_pdm_collision_distance: float = 2.5
    b2d_pdm_ttc_safe_distance: float = 8.0
    b2d_pdm_ttc_mode: str = "pad"  # "pad" uses box extrapolation; "distance" uses center distance.
    b2d_pdm_wrong_lane_cos_threshold: float = 0.0
    b2d_pdm_wrong_lane_allow_from_gt: bool = True
    b2d_pdm_gt_wrong_lane_allow_threshold: float = 0.95
    b2d_pdm_wrong_lane_allowed_floor: float = 0.70
    b2d_pdm_wrong_lane_strict_power: float = 2.0
    b2d_pdm_ego_half_length: float = 2.042
    b2d_pdm_ego_half_width: float = 0.925
    b2d_pdm_ego_rear_axle_to_center: float = 0.39
    b2d_pdm_comfort_mode: str = "relative"  # "relative", "pad_binary", or "threshold".
    b2d_pdm_comfort_accel_floor: float = 0.5
    b2d_pdm_comfort_yaw_rate_floor: float = 0.1
    b2d_pdm_comfort_accel_threshold: float = 4.5
    b2d_pdm_comfort_yaw_rate_threshold: float = 1.2
    pdm_score_risk_area_enable: bool = False  # enable PAD-style scorer auxiliary head for static offroad-risk only
    pdm_score_risk_area_weight: float = 0.0  # weight of scorer offroad-risk auxiliary BCE loss
    grpo_enable: bool = False  # enable mode-level GRPO fine-tuning over current proposals
    grpo_weight: float = 0.0  # weight for GRPO policy loss
    grpo_policy_source: str = "pdm_score"  # policy logits: "pdm_score", "poses_cls", "generalist", or "specialist"
    grpo_reward_topk: int = 0  # score only top-k modes by policy logits (0 = all modes)
    grpo_clip_eps: float = 0.2  # PPO/GRPO ratio clipping epsilon
    grpo_temp: float = 1.0  # temperature for policy log-softmax
    grpo_entropy_weight: float = 0.0  # entropy bonus weight inside GRPO loss
    grpo_min_reward_std: float = 1e-3  # skip samples whose group reward std is too small
    grpo_detach_reward: bool = True  # detach group-normalized rewards/advantages
    pdm_val_use_online_score: bool = False  # log DrivoR-style val PDM metrics
    pdm_val_score_topk: int = 0  # optionally score only top-k proposals in val (0=all)
    pdm_val_metrics_use_inference_forward: bool = False  # compute val/monitor PDM metrics from inference-style forward(features)
    pdm_val_score_use_selected_trajectory: bool = False  # score only selected trajectory (predictions['trajectory']) to align with run_pdm_score
    pdm_val_skip_primary_loader: bool = False  # 涓嶄娇鐢╲al鏁版嵁闆?skip online PDM scoring on primary val loader (dataloader_idx=0)
    pdm_score_output_infraction_details: bool = False  # export collision/offroad diagnostic tensors from online PDM scoring
    pdm_score_output_infraction_in_train: bool = True  # enable infraction diagnostics during training steps
    pdm_score_output_infraction_in_val: bool = True  # enable infraction diagnostics during validation/inference steps
    # Hard-case uncertainty recording based on specialist adapter uncertainty.
    hardcase_record_enable: bool = False  # expose adapter-uncertainty diagnostics from inference predictions
    hardcase_threshold: float = 0.55  # threshold on adapter uncertainty used to mark hard cases
    hardcase_region_bins_deg: Tuple[float, float, float, float] = (
        -50.0,
        -15.0,
        15.0,
        50.0,
    )  # 5 regions: hard-right / right / straight / left / hard-left
    hardcase_dispersion_ref_m: float = 8.0  # normalize endpoint dispersion by this distance (meters)
    hardcase_safety_component_ids: Tuple[int, int, int] = (
        0,
        1,
        3,
    )  # component ids used for safety proxy: collision, DAC, TTC
    hardcase_u_weight_entropy: float = 0.30  # legacy mixed-U config, unused
    hardcase_u_weight_margin: float = 0.25  # legacy mixed-U config, unused
    hardcase_u_weight_region_entropy: float = 0.15  # legacy mixed-U config, unused
    hardcase_u_weight_dispersion: float = 0.15  # legacy mixed-U config, unused
    hardcase_u_weight_safety_uncertainty: float = 0.10  # legacy mixed-U config, unused
    hardcase_u_weight_safety_risk: float = 0.05  # legacy mixed-U config, unused
    hardcase_u_weight_adapter_variance: float = 0.0  # legacy mixed-U config, unused
    hardcase_u_weight_attn_entropy: float = 0.0  # legacy mixed-U config, unused
    hardcase_u_weight_attn_variance: float = 0.0  # legacy mixed-U config, unused
    hardcase_adapter_uncertainty_ref: float = 0.05  # normalization scale for selected-mode adapter variance
    # Scene+proposal-based hardcase gating with residual scorer head.
    hardcase_gate_enable: bool = False  # enable hardcase gate and residual scorer branch
    hardcase_gate_use_scene_feature: bool = True  # use scene-level feature in hardcase gate
    hardcase_gate_use_proposal_feature: bool = True  # use proposal geometry feature in hardcase gate
    hardcase_gate_hidden_dim: int = 256  # hidden dim of hardcase gate MLP
    hardcase_gate_temperature: float = 1.0  # temperature applied before sigmoid for gate probability
    hardcase_gate_threshold: float = 0.65  # inference hard trigger threshold on gate probability
    hardcase_gate_soft_in_train: bool = True  # use soft gate probability during training
    hardcase_gate_residual_scale: float = 0.5  # residual score scale when hard gate is active
    hardcase_gate_prior: float = 0.10  # expected hardcase ratio prior for gate regularization
    hardcase_gate_prior_weight: float = 0.0  # weight of prior regularization on gate activation
    hardcase_gate_residual_reg_weight: float = 0.0  # weight of residual penalty outside hard region
    hardcase_gate_train_only: bool = False  # freeze generalist and train only hardcase gate/residual scorer branch
    hardcase_gate_supervision_enable: bool = False  # supervise selected-mode gate with score+BEV hard label
    hardcase_gate_supervision_weight: float = 0.0  # weight of gate supervision BCE
    hardcase_gate_supervision_score_thresh: float = 0.55  # hard if selected score (sigmoid logit) is below this
    hardcase_gate_supervision_score_temp: float = 0.10  # temperature for score-to-risk sigmoid
    hardcase_gate_supervision_bev_z_thresh: float = 0.5  # hard if per-sample BEV CE z-score exceeds this
    hardcase_gate_supervision_bev_temp: float = 0.5  # temperature for BEV z-score-to-risk sigmoid
    hardcase_gate_supervision_combine: str = "max"  # combine score/bev risks: max or mean
    # Score residual specialist (R2SE-triggered), no learned trigger.
    hardcase_score_residual_enable: bool = False  # enable specialist residual branch on PDM score head
    hardcase_score_residual_use_scene_feature: bool = True  # fuse scene feature for residual score
    hardcase_score_residual_use_proposal_feature: bool = True  # fuse proposal geometry feature for residual score
    hardcase_score_residual_use_mode_attention_feature: bool = False  # fuse mode-bev-attention context into residual score head only
    hardcase_score_residual_hidden_dim: int = 256  # hidden dim of specialist residual score MLP
    hardcase_score_residual_scale: float = 0.5  # residual score scale
    hardcase_score_residual_loss_weight: float = 0.5  # extra PDM loss weight on specialist score logits
    hardcase_score_residual_hard_filter_enable: bool = True  # train specialist score only on hard samples
    hardcase_score_residual_use_score_risk: bool = True  # use generalist selected-score risk to define hard samples
    hardcase_score_residual_use_oracle_score_risk: bool = False  # use online PDM true score (selected trajectory) for score risk
    hardcase_score_residual_use_bev_risk: bool = True  # use BEV CE z-score risk to define hard samples
    hardcase_score_residual_score_thresh: float = 0.55  # hard when selected score is below this
    hardcase_score_residual_score_temp: float = 0.10  # temperature for score risk sigmoid
    hardcase_score_residual_bev_z_thresh: float = 0.5  # hard when BEV CE z-score is above this
    hardcase_score_residual_bev_temp: float = 0.5  # temperature for BEV risk sigmoid
    hardcase_score_residual_combine: str = "max"  # combine score/bev risks: max or mean
    hardcase_score_residual_hard_target_thresh: float = 0.6  # target-risk threshold to mark hard samples
    hardcase_score_residual_easy_target_risk_thresh: float = 0.30  # easy if target-risk is below this threshold
    hardcase_score_residual_easy_selected_score_thresh: float = 0.90  # easy if selected generalist score is above this threshold
    hardcase_score_residual_easy_require_high_score: bool = False  # require high selected score to include easy sample
    hardcase_score_residual_easy_consistency_weight: float = 0.0  # weight for keeping specialist score close to generalist on easy samples
    hardcase_score_residual_easy_distribution_weight: float = 0.0  # weight for matching specialist/generalist mode-score distribution on easy samples
    hardcase_score_residual_easy_uncertainty_weight: float = 0.0  # weight for suppressing specialist uncertainty on easy samples
    hardcase_score_residual_diverge_weight: float = 0.0  # optional hardcase loss to enlarge specialist/generalist score gap
    hardcase_score_residual_diverge_margin: float = 0.15  # target minimum selected-score gap on hard samples
    hardcase_score_residual_train_only: bool = False  # freeze generalist and train only score residual branch
    hardcase_score_residual_lora_enable: bool = False  # use K parallel LoRA adapters for residual score branch
    hardcase_score_residual_lora_num_adapters: int = 4  # number of parallel residual LoRA adapters (K)
    hardcase_score_residual_lora_rank: int = 8  # LoRA rank for each adapter
    hardcase_score_residual_lora_alpha: float = 8.0  # LoRA alpha; effective scale is alpha/rank
    hardcase_score_residual_lora_dropout: float = 0.0  # dropout before LoRA adapters
    hardcase_score_residual_lora_init_scale: float = 1e-3  # init std for LoRA A matrix (B starts at zero)
    hardcase_score_residual_adapter_independent_loss_enable: bool = True  # supervise each residual adapter independently to avoid ensemble collapse
    hardcase_score_residual_adapter_independent_loss_weight: float = 0.5  # relative weight for per-adapter specialist supervision
    hardcase_score_residual_adapter_decor_weight: float = 0.0  # hard-only decorrelation regularization across LoRA adapters
    hardcase_score_residual_adapter_decor_eps: float = 1e-6  # epsilon for adapter decorrelation normalization
    # Generalist + specialist residual correction branch (R2SE-style routing uses this specialist at inference).
    hardcase_specialist_enable: bool = False  # enable residual specialist branch on top of generalist trajectories
    hardcase_specialist_num_heads: int = 2  # number of residual expert heads
    hardcase_specialist_hidden_dim: int = 256  # hidden dim of specialist residual MLPs
    hardcase_specialist_use_status: bool = True  # fuse ego/status token into specialist branch
    hardcase_specialist_delta_scale: float = 0.5  # scale of residual correction
    hardcase_specialist_delta_clip_xy: float = 1.0  # clip xy residual (meters)
    hardcase_specialist_delta_clip_heading: float = 0.1  # clip heading residual (radians)
    hardcase_specialist_loss_weight: float = 0.2  # weight of specialist supervision loss
    hardcase_specialist_hard_target_thresh: float = 0.6  # target-risk threshold to mark hard samples
    hardcase_specialist_use_score_risk: bool = True  # use selected trajectory score to build hard target
    hardcase_specialist_use_bev_risk: bool = True  # use BEV CE z-score to build hard target
    hardcase_specialist_nonhard_consistency_weight: float = 0.05  # keep specialist close to generalist on non-hard samples
    hardcase_specialist_delta_reg_weight: float = 0.0  # optional L1 regularization on specialist residual magnitude
    # R2SE-style statistical hardcase routing (non-learned trigger).
    hardcase_r2se_enable: bool = False  # enable uncertainty-based specialist/generalist switching at inference
    hardcase_r2se_disable_learned_gate: bool = True  # force-disable learned hardcase gate when R2SE routing is enabled
    hardcase_r2se_u_key: str = "u_adapter_variance"  # uncertainty key from _extract_hardcase_signals used by GPD switch
    hardcase_r2se_sigma: float = 0.75  # confidence threshold sigma in Eq.14-style switch
    hardcase_r2se_gpd_param_path: Optional[str] = None  # JSON with {u0, shape, scale} fitted from hard cases
    hardcase_r2se_force_specialist: bool = False  # diagnostic override: always switch to specialist during inference
    hardcase_r2se_fallback_policy: str = "poses_cls"  # generalist fallback mode: poses_cls | first_mode
    hardcase_r2se_apply_fallback: bool = True  # apply fallback trajectory when PGPD(U)<=sigma
    hardcase_r2se_debug_record: bool = True  # append r2se_* debug metrics to get_last_inference_debug
    # 鎺ㄧ悊澶氭閲囨牱anchor锛岀粺涓€鎵撳垎
    inference_multisample_enable: bool = False  # test-time random resampling and reranking
    inference_multisample_count: int = 6  # 澶氭鏈夋晥鏋渘umber of stochastic forward samples when multisample is enabled
    inference_multisample_force_anchor_sample: bool = True  # force learned Gaussian anchor sampling in eval for multisample
    inference_dedup_enable: bool = False  # 鍘婚噸娌℃湁鏁堟灉deduplicate proposals before final reranking in inference
    inference_dedup_mean_l2_thresh_m: float = 0.6  # duplicate if mean L2 distance across trajectory is below this threshold
    inference_dedup_endpoint_thresh_m: float = 1.2  # duplicate if endpoint distance is below this threshold
    inference_dedup_heading_thresh_deg: float = 15.0  # <=0 disables heading constraint for duplicate check
    inference_dedup_max_modes: int = 0  # keep at most this many unique modes (0 uses all available modes)
    # 杞婚噺澶氭 refinement锛堥粯璁ゅ叧闂級
    refine_enable: bool = False  # enable iterative residual refinement on predicted trajectories
    refine_steps: int = 2  # number of refinement iterations
    refine_hidden_dim: int = 256  # hidden dim for refinement MLP
    refine_use_ego_context: bool = True  # fuse ego/status token when refining trajectories
    refine_delta_clip_xy: float = 1.5  # per-step clip for xy residual (meters)
    refine_delta_clip_heading: float = 0.2  # per-step clip for heading residual (radians)
    refine_aux_loss_weight: float = 0.0  # optional auxiliary loss weight for intermediate refine steps
    refine_aux_gt_weight: float = 1.0  # GT distance term weight in refine auxiliary loss
    refine_aux_candidate_weight: float = 1.0  # candidate distance term weight in refine auxiliary loss
    refine_aux_use_candidates: bool = True  # include candidate-trajectory supervision in refine auxiliary loss
    refine_aux_candidate_use_score_filter: bool = True  # keep only candidates with pdm_score > gt_pdm_score for refine auxiliary loss
    mode_bev_attention_use_scorer_pre: bool = False  # inject mode attention context before scorer decoder
    mode_bev_attention_use_scorer_post: bool = True  # inject mode attention context after scorer decoder
    mode_bev_attention_enable: bool = False  # enable trajectory-centered BEV attention extraction
    mode_bev_attention_use_refine: bool = True  # feed mode attention context to refine module
    mode_bev_attention_use_scorer: bool = True  # feed mode attention context to scorer head
    mode_bev_attention_recompute_after_refine: bool = True  # recompute attention with refined trajectories for scorer
    mode_bev_attention_refine_recompute_each_step: bool = False  # recompute attention at every refine step
    mode_bev_attention_temperature: float = 1.0  # softmax temperature for mode attention map
    mode_bev_attention_risk_enable: bool = False  # inject BEV risk-region prior into mode attention logits
    mode_bev_attention_risk_use_target_in_train: bool = True  # use GT BEV labels to build risk prior during training
    mode_bev_attention_risk_bias_weight: float = 2.0  # scale of risk prior bias added to mode attention logits
    mode_bev_attention_risk_offroad_weight: float = 1.0  # weight for non-drivable/offroad risk
    mode_bev_attention_risk_obstacle_weight: float = 0.5  # weight for dynamic obstacle risk
    mode_bev_attention_risk_include_pedestrian: bool = True  # include pedestrian BEV class in obstacle risk prior
    mode_bev_attention_risk_include_static: bool = False  # include static-obstacle BEV class in obstacle risk prior
    mode_bev_attention_risk_static_weight: float = 0.5  # weight for static obstacle risk
    mode_bev_attention_aux_enable: bool = True  # enable attention-map auxiliary supervision from PDM infraction signals
    mode_bev_attention_aux_weight: float = 0.2  # weight of attention auxiliary KL loss
    mode_bev_attention_aux_sigma_px: float = 2.0  # gaussian sigma in BEV pixels for point supervision
    mode_bev_attention_aux_time_decay_enable: bool = True  # decay auxiliary attention target by event time
    mode_bev_attention_aux_time_decay_tau_s: float = 2.0  # exponential decay tau (seconds)
    mode_bev_attention_aux_time_decay_min: float = 0.1  # lower bound for time-decay weight
    # 鎺ㄧ悊鍚庡鐞嗭紝瀵筆DM缁勪欢澶氭鎵撳垎鎼滅储
    inference_pdm_postprocess_enable: bool = False  # 鍙傛暟鎼滅储娌℃湁鏁堟灉apply inference-only score calibration on PDM components
    inference_pdm_component_weights: Tuple[float, float, float, float, float, float] = (
    1.2,
    1.2,
    1.1,
    1.0,
    0.9,
    1.2,
    )  # inference-only aggregation weights for [noc, dac, progress, ttc, comfort, ddc]
    inference_pdm_component_bias: Tuple[float, float, float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    )  # inference-only additive bias on component logits before aggregation
    inference_pdm_logit_temperature: float = 0.9  # inference-only logit temperature on component logits
    inference_pdm_use_logsigmoid_aggregate: bool = True  # inference-only aggregation form
    
    checkpoint_monitor: str = "val/trajectory_loss_epoch"  # metric to track best ckpt
    checkpoint_mode: str = "min"  # min or max
    checkpoint_save_top_k: int = 2  # number of best ckpts to keep
    checkpoint_save_last: bool = False  # also save last ckpt
    checkpoint_save_last_on_train_epoch_end: bool = True  # save last.ckpt from train epoch end instead of waiting for validation
    checkpoint_last_every_n_epochs: int = 1  # frequency for the train-epoch last.ckpt callback
    checkpoint_last_every_n_train_steps: int = 0  # >0 saves last.ckpt by train steps instead of epochs
    checkpoint_filename: str = "best"  # filename for best ckpt
    save_best_bev_semantic_ckpt: bool = False  # save best_bev checkpoint

    include_ego_history: bool = True  # cache ego history poses in features     
    include_velocity_bev: bool = True  # cache BEV velocity field (vx, vy) between prev/current
    include_future_bev_semantic_map: bool = True  # cache future BEV semantic map aligned to current frame
    ego_history_to_status: bool = False  # whether to add ego history encoding to status_encoding

    anchor_free: bool = True  # build anchors from feasible area instead of fixed kmeans
    anchor_free_use_target: bool = True  # prefer target feasible mask when available
    anchor_use_feasible_area_mask: bool = True  # constrain anchor initialization by feasible_area_mask
    anchor_free_step_meters: float = 5.0  # step size (or Gaussian sigma) in meters
    anchor_free_step_use_speed: bool = True  # use ego speed to set anchor step size
    anchor_free_step_min_m: float = 1.0  # min step when using speed-based steps
    anchor_free_step_max_m: float = 6.0  # max step when using speed-based steps
    anchor_free_forward_only: bool = True  # constrain anchors to forward direction in BEV
    anchor_free_gaussian: bool = True  # sample anchors with Gaussian noise in feasible area
    anchor_free_gaussian_temporal_enable: bool = False  # use temporally correlated constrained Gaussian anchor generation
    anchor_free_gaussian_temporal_rho: float = 0.85  # AR(1) correlation for Gaussian noise across trajectory steps
    anchor_free_gaussian_lateral_ratio: float = 0.35  # sigma_lat = ratio * sigma_parallel
    anchor_free_gaussian_step_growth: float = 0.08  # per-step growth factor of Gaussian std
    anchor_free_gaussian_correction_iters: int = 2  # projection attempts when sampled point is infeasible
    anchor_free_gaussian_correction_blend: float = 0.5  # blend with target when correcting infeasible samples
    anchor_free_skip_diffusion_noise: bool = True  # skip add_noise when anchor_free is on
    anchor_free_max_seed_offset_m: float = 2.0  # 0 disables; prefer seeds within this radius
    anchor_free_forward_pad_m: float = 11.0  # pad feasible mask forward (meters) for anchor generation
    
    anchor_learned_gaussian_enable: bool = True  # predict Gaussian(mu, sigma) by network for anchor generation
    anchor_learned_gaussian_use_physics_prior: bool = True  # add network residual on top of physics prior mean
    anchor_learned_gaussian_residual_scale: float = 0.2  # scale for network mean residual
    anchor_learned_gaussian_min_std_m: float = 0.2  # minimum Gaussian std in meters
    anchor_learned_gaussian_max_std_m: float = 8.0  # maximum Gaussian std in meters
    anchor_learned_gaussian_step_growth: float = 0.05  # per-step growth for predicted std
    anchor_learned_gaussian_sample_in_eval: bool = False  # sample at eval; otherwise use mean
    anchor_learned_gaussian_nll_weight: float = 0.2  # supervision weight for GT NLL under predicted Gaussian
    anchor_learned_gaussian_reg_weight: float = 1.0  # supervision weight for nearest-mode mean regression
    anchor_learned_gaussian_candidate_weight: float = 0.2  # auxiliary NLL supervision weight from high-quality candidates
    anchor_learned_gaussian_candidate_use_score_filter: bool = True  # keep only candidates with pdm_score > gt_pdm_score
    anchor_visualize_enable: bool = True  # save anchor visualization images during train/test
    anchor_visualize_dir: str = "outputs/anchor_vis"  # output directory for anchor visualizations
    anchor_visualize_every_n_forward: int = 1000  # save one visualization every N forward calls (<=0 means 1)
    anchor_visualize_max_files: int = 10  # max number of visualization files to save (<=0 means unlimited)
    anchor_visualize_batch_index: int = 0  # which batch item to visualize
    anchor_visualize_plot_mu: bool = True  # plot learned Gaussian mean trajectories when available
    anchor_visualize_plot_std: bool = True  # plot simple sigma indicator at final step when available
    anchor_visualize_plot_gt: bool = True  # plot GT trajectory when available
    anchor_visualize_plot_candidates: bool = True  # plot candidate trajectories when available
    
    reachability_use_bicycle: bool = True  # use bicycle model to compute reachable mask
    reachability_use_for_anchor: bool = True  # apply reachability mask for anchor sampling
    reachability_output_feature: bool = True  # output reachability feature vector
    reachability_use_history: bool = True  # use ego history to estimate initial heading/speed
    reachability_use_history_speed: bool = False  # override speed with history estimate
    reachability_history_steps: int = 2  # number of history steps to estimate heading
    reachability_history_min_dist_m: float = 0.5  # min displacement to trust heading
    reachability_wheel_base_m: float = 3.089  # vehicle wheelbase in meters
    reachability_max_steer_deg: float = 15.0  # max steering angle in degrees
    reachability_min_accel_mps2: float = -1.0  # min longitudinal accel (m/s^2)
    reachability_max_accel_mps2: float = 1.0  # max longitudinal accel (m/s^2)
    reachability_steer_samples: int = 7  # number of steering samples
    reachability_accel_samples: int = 3  # number of accel samples
    reachability_horizon_steps: int = 0  # 0 uses trajectory_sampling.num_poses
    reachability_use_speed: bool = True  # use ego speed as initial velocity
    reachability_anchor_strict_steps: int = 3  # strict steps before relaxing to BEV feasible area
    anchor_relaxed_allow_vehicle_overlap: bool = True  # after strict steps, allow anchors to pass current vehicle-occupied cells
    anchor_relaxed_use_target_vehicle_mask: bool = True  # in train, prefer GT BEV semantic map to extract vehicle mask
    anchor_relaxed_include_pedestrians: bool = False  # also relax pedestrian-occupied cells when true
    reachability_dilate_m: float = 3.0  # expand reachability mask by this meters

    # optmizer
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type = "AdamW"
    scheduler_type = "MultiStepLR"
    cfg_lr_mult = 0.5
    opt_paramwise_cfg = {
        "name":{
            "image_encoder":{
                "lr_mult": cfg_lr_mult
            }
        }
    }
    # optimizer=dict(
    #     type="AdamW",
    #     lr=1e-4,
    #     weight_decay=1e-6,
    # )
    # scheduler=dict(
    #     type="MultiStepLR",
    #     milestones=[90],
    #     gamma=0.1,
    # )

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
