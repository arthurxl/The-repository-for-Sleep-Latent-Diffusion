from generative.networks.nets import AutoencoderKL, PatchDiscriminator

autoencoder = AutoencoderKL(spatial_dims=1, in_channels=1, out_channels=1,
                            num_channels=(2, 4), latent_channels=3, num_res_blocks=2,
                            norm_num_groups=1, attention_levels=(False, False),
                            with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)

patchdiscriminator = PatchDiscriminator(spatial_dims=1, num_layers_d=3, num_channels=64,
                                        in_channels=1, out_channels=1, kernel_size=3,
                                        norm="BATCH", bias=False, padding=1)
