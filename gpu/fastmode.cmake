# SPDX-License-Identifier: MIT
# What the user's "Fast mode" changes, per shader source: the defines of its _lt kernels. docs/FASE24-FASTMODE.md has the
# ladder. A kernel left out here stays on its _f file in that mode.
set(LT_DEFS_enc0_stage_32_64 LT_FAR=2 LT_FAR1=2 LT_Q16=1 LT_VO=1 LT_SYM=1 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_up_32_64_64 LT_FAR=3 LT_FAR1=3 LT_Q16=1 LT_VO=1 LT_SYM=1 LT_UP8=1 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_head_32_64 LT_FAR=3 LT_FAR1=3 LT_Q16=1 LT_VO=1 LT_SYM=1 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_enc0_attr LT_ATTR_TAB=1 LT_ATTR_LATE=1 LT_ATTR_D9=1)
set(LT_DEFS_post LT_POST_PAIR=1 LT_POST_HW=1 LT_POST_LDS=1 LT_POST_EXPO=1)
set(LT_DEFS_post_dlt LT_POST_PAIR=1 LT_DLT_PAIR=1 LT_POST_HW=1 LT_POST_LDS=1 LT_POST_EXPO=1)
set(LT_DEFS_pooled_history2 LT_PH2_LOG=1)
set(LT_DEFS_downsample LT_DS_LOAD=1 LT_DS_FAST=1)
set(LT_DEFS_enc_stage_64_64_96 LT_MI8=1 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_enc_stage_96_128_128 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_enc_stage_128_128_160 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_enc_stage_160_256_160 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_enc_stage_160_256_0 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_up_160_256_160 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_dfn_160_256 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_up_128_128_160 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_dfn_128_128 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_up_96_128_128 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_dfn_96_128 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_up_64_64_96 LT_UP8=1 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_dec_dfn_64_64 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_DEFS_downsample_tl LT_DS_LOAD=1 LT_DS_FAST=1)
set(LT_DEFS_dfn_export LT_DFN_LOAD=1)
set(LT_DEFS_dfn_import LT_DFN_LOAD=1)
# The deep levels from an earlier frame while the mode is on (Engine::set_deep): "mode,level,period,guard". Empty: off.
# "1,5,2,4": enc5 every other frame; "1,4,2,4": enc4, enc5, dec4_0 and dec4_1 every other frame (4 frames after a reset run whole).
# The experimental level above it (OnrRrSetFastMode 2): these kernels again as _lx, the rest of the mode as it is, and the
# deep levels read from the previous frame. About 8% off the frame at Quality for some 5 dB.
set(LX_DEFS_enc0_stage_32_64 LT_FAR=2 LT_FAR1=2 LT_Q16=1 LT_VO=1 LT_SYM=1 LT_GELU=0.31 LT_GELU_PART=1 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LX_DEFS_dec_up_32_64_64 LT_FAR=3 LT_FAR1=3 LT_Q16=1 LT_VO=1 LT_SYM=1 LT_GELU=0.31 LT_UP8=1 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LX_DEFS_dec_head_32_64 LT_FAR=2 LT_FAR1=2 LT_Q16=1 LT_VO=1 LT_SYM=1 LT_GELU=0.31 LT_EXP=1 LT_RCP=1 LT_HM=1)
set(LT_ENGINE_DEEP "2,4,2,4")
