import k_diffusion as K
import torch

def make_cond_model_fn(model, cond_fn):
    def cond_model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return cond_model_fn

def sample_k(
        model_fn, 
        noise=None, 
        init_data=None,
        mask=None,
        steps=100, 
        sampler_type="dpmpp-2m-sde", 
        sigma_min=0.5, 
        sigma_max=50, 
        rho=1.0, device="cuda", 
        callback=None, 
        cond_fn=None,
        return_all_latents=False,
        **extra_args
    ):
    
    # print(extra_args.keys())

    denoiser = K.external.VDenoiser(model_fn)

    if cond_fn is not None:
        denoiser = make_cond_model_fn(denoiser, cond_fn)

    # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
    sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
    
    # Scale the initial noise by sigma 
    if noise is not None:
        noise_init = noise * sigmas[0]

    wrapped_callback = callback

    if mask is None and init_data is not None:
        # VARIATION (no inpainting)
        # set the initial latent to the init_data, and noise it with initial sigma
        x = init_data 
        # return x
    else:
        # SAMPLING
        # set the initial latent to noise
        x = noise_init
        
    # dse = laion_clap.CLAP_Module(enable_fusion=False)
    # dse.load_ckpt()

    with torch.cuda.amp.autocast():
        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args, return_all_latents=return_all_latents)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            # return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args, controller=controller)
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=wrapped_callback, extra_args=extra_args)


@torch.no_grad()
def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        padded_phone_ids=None,
        prosody_cond=None,
        mask_positions=None,
        padding_mask=None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        sigma_mask_end=None,
        filter_sigma=0.1,
        init_noise_level: float = 1.0,
        k_round=1,
        mask_args: dict = None,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """
    
    # The length of the output in audio samples 
    audio_sample_size = sample_size

    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
    
    print(model.pretransform.downsampling_ratio, " downsampling ratio")
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1, dtype=np.uint32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    # noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)
    noise = torch.randn([1, model.io_channels, sample_size], device=device)
    noise = noise.repeat(batch_size, 1, 1)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
            
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    mask_callback = None
    if init_audio is not None:
        # The user supplied some initial audio (for inpainting or variation). Let us prepare the input audio.
        in_sr, init_audio = init_audio
        
        init_audio_time = init_audio.clone()
        call_back_obj = MaskCallback(init_audio=init_audio_time,
                                     sigma_stop=sigma_mask_end,
                                     fs=in_sr,
                                     pretransform=model.pretransform,
                                     filter_sigma=filter_sigma,
                                     noise=noise)
        
        io_channels = model.io_channels

        # For latent models, set the io_channels to the autoencoder's io_channels
        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        # Prepare the initial audio for use by the model
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        # For latent models, encode the initial audio into latents
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)
    else:
        # The user did not supply any initial audio for inpainting or variation. Generate new output from scratch. 
        init_audio = None
        init_noise_level = None
        mask_args = None

    # Inpainting mask
    if init_audio is not None and mask_args is not None:
        # Cut and paste init_audio according to cropfrom, pastefrom, pasteto
        # This is helpful for forward and reverse outpainting
        cropfrom = math.floor(mask_args["cropfrom"]/100.0 * sample_size)
        pastefrom = math.floor(mask_args["pastefrom"]/100.0 * sample_size)
        pasteto = math.ceil(mask_args["pasteto"]/100.0 * sample_size)
        assert pastefrom < pasteto, "Paste From should be less than Paste To"
        croplen = pasteto - pastefrom
        if cropfrom + croplen > sample_size:
            croplen = sample_size - cropfrom 
        cropto = cropfrom + croplen
        pasteto = pastefrom + croplen
        cutpaste = init_audio.new_zeros(init_audio.shape)
        cutpaste[:, :, pastefrom:pasteto] = init_audio[:,:,cropfrom:cropto]
        #print(cropfrom, cropto, pastefrom, pasteto)
        init_audio = cutpaste
        # Build a soft mask (list of floats 0 to 1, the size of the latent) from the given args
        mask = build_mask(sample_size, mask_args)
        mask = mask.to(device)
    elif init_audio is not None and mask_args is None:
        # variations
        sampler_kwargs["sigma_max"] = init_noise_level
        mask = None 
    else:
        mask = None

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!
    
    
    diff_objective = model.diffusion_objective

    if diff_objective == "v":    
        # check if init_noise_level is List
        if not isinstance(init_noise_level, list):
            init_noise_level = [init_noise_level] * k_round
            
        if not isinstance(sigma_mask_end, list):
            sigma_mask_end = [sigma_mask_end] * k_round
        
        assert len(sigma_mask_end) == k_round, "sigma_mask_end should be a list of length k_round"
        assert len(init_noise_level) == k_round, "init_noise_level should be a list of length k_round"
        # k-diffusion denoising process go!
        sampled = init_audio


        # sampler_kwargs["sigma_max"] = init_noise
        # call_back_obj.sigma_stop = mask_end
        sampled = sample_k(model, 
                        noise, 
                        sampled,
                        mask, 
                        steps, 
                        **sampler_kwargs, 
                        **conditioning_inputs, 
                        **negative_conditioning_tensors, 
                        cfg_scale=cfg_scale, 
                        batch_cfg=True, 
                        rescale_cfg=True,
                        device=device,)

    # v-diffusion: 
    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled