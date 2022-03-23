#train prompt using positive with ious in [0,5,0.6] and negative proposals
python prompt/run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_5_6_end soft 0.5 0.5 0.6 8 end
#train prompt using positive with ious in [0,6,0.7] and negative proposals
python prompt/run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_6_7_end soft 0.5 0.6 0.7 8 end
#train prompt using positive with ious in [0,7,0.8] and negative proposals
python prompt/run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_7_8_end soft 0.5 0.7 0.8 8 end
#train prompt using positive with ious in [0,8,0.9] and negative proposals
python prompt/run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_8_9_end soft 0.5 0.8 0.9 8 end
#train prompt using positive with ious in [0,9,1.] and negative proposals
python prompt/run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_9_10_end soft 0.5 0.9 1.1 8 end
#ensemble
python prompt/run.py test data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_10_end soft 0.0 0.5 0.5 checkpoints/exp/fg_bg_5_5_6_endepoch6.pth checkpoints/exp/fg_bg_5_6_7_endepoch6.pth checkpoints/exp/fg_bg_5_7_8_endepoch6.pth checkpoints/exp/fg_bg_5_8_9_endepoch6.pth checkpoints/exp/fg_bg_5_9_10_endepoch6.pth