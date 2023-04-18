`mc_rtt_<{tune, token}>`: In the Unsup PT -> Sup PT -> sup FT line
`mc_rtt_decode_<{tune, token}>`: In the Sup PT -> sup FT line
`mc_rtt_joint_<{tune, token}>`: In Joint PT -> sup FT line

`mc_rtt_joint_unsup_<{tune, token}>`: In Joint PT -> unsupervised FT line

- Tune refers to full tuning with decay (decay not swept, taken from MAE Kaiming paper)
- Token is token only. Perhaps not a priority at the moment.