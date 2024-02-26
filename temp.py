from inference_VQ_Diffusion import VQ_Diffusion

# captions = ['a bird with large white tail']

with open('example_captions.txt') as file:
    captions = file.readlines()

# pretrained model
VQ_Diffusion_model = VQ_Diffusion(
    config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/cub_pretrained.pth')
for caption in captions:
    VQ_Diffusion_model.inference_generate_sample_with_condition(
        caption, truncation_rate=0.86, save_root="RESULT_PRETRAINED", batch_size=1)

# sampling
VQ_Diffusion_model = VQ_Diffusion(
    config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/cub200_train/cub_best_model.pth')
for caption in captions:
    VQ_Diffusion_model.inference_generate_sample_with_condition(
        caption, truncation_rate=0.86, save_root="RESULT", batch_size=1)
