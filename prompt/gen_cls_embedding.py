import coop_mini
from classname import *
# from trainer import checkpoint
import sys, torch

_, prompt_path, save_name, dataset = sys.argv

def checkpoint(model, name, class_names):
    with torch.no_grad():
        prompts, tokenized_prompts = model.prompt_learner.forward_for_classes(class_names)
        text_features = model.text_encoder(prompts, tokenized_prompts)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # text_features = torch.cat([text_features], dim=0)

        text_features = text_features[None, ...]
        torch.save(text_features, name+".pth")
        # print(prompts.shape)
        # print(text_features.shape)
        prompt = model.prompt_learner.ctx.data
        torch.save(prompt, name+"_prompt.pth")
clip_model = coop_mini.load_clip_to_cpu().float()
for params in clip_model.parameters():
    params.requires_grad_(False)

prompt = torch.load(prompt_path)
if dataset == 'obj365':
    model = coop_mini.CustomCLIP(Objects365_CLASSES, clip_model, False, prompt).to('cuda')
    checkpoint(model, save_name, Objects365_CLASSES)
elif dataset == 'voc':
    model = coop_mini.CustomCLIP(VOC_CLASSES, clip_model, False, prompt).to('cuda')
    checkpoint(model, save_name, VOC_CLASSES)
elif dataset == 'coco':
    model = coop_mini.CustomCLIP(COCO_CLASSES, clip_model, False, prompt).to('cuda')
    checkpoint(model, save_name, COCO_CLASSES)
