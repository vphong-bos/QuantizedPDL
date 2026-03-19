from aimet_torch import quantsim, bias_correction

def apply_bias_correction(
    model,
    calib_loader,
    image_height,
    image_width,
    quant_scheme,
    default_param_bw,
    default_output_bw,
    config_file,
    bias_corr_num_quant_samples,
    bias_corr_num_bias_samples,
    bias_corr_empirical_only = True,

):
    model = model.cpu().eval()

    params = quantsim.QuantParams(
        weight_bw=default_param_bw,
        act_bw=default_output_bw,
        round_mode="nearest",
        quant_scheme=quant_scheme,
        config_file=config_file,
    )

    conv_bn_dict = None
    if not bias_corr_empirical_only:
        conv_bn_dict = bias_correction.find_all_conv_bn_with_activation(
            model,
            input_shape=(1, 3, image_height, image_width),
        )

    bias_correction.correct_bias(
        model,
        params,
        num_quant_samples=min(bias_corr_num_quant_samples, len(calib_loader.dataset)),
        data_loader=calib_loader,
        num_bias_correct_samples=min(bias_corr_num_bias_samples, len(calib_loader.dataset)),
        conv_bn_dict=conv_bn_dict,
        perform_only_empirical_bias_corr=bias_corr_empirical_only,
    )

    return model