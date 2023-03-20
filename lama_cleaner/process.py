#!/usr/bin/env python3

from lama_cleaner.helper import (
    load_img,
    resize_max_size,
)
import multiprocessing
import os
import random
import time
from typing import Union
from PIL import Image, ImageOps

import cv2
import torch
import numpy as np
from loguru import logger

from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config


def diffuser_callback(i, t, latents):
    pass
    # socketio.emit('diffusion_step', {'diffusion_step': step})


def load_img(image, gray: bool = False, return_exif: bool = False):
    alpha_channel = None

    try:
        if return_exif:
            exif = image.getexif()
    except:
        exif = None
        logger.error("Failed to extract exif from image")

    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_exif:
        return np_img, alpha_channel, exif
    return np_img, alpha_channel


def process(original_image, original_mask, model_name):
    input = {
    }

    model = ModelManager(
        name=model_name,
        device=torch.device('cuda'),
        # no_half=args.no_half,
        # hf_access_token=args.hf_access_token,
        # disable_nsfw=args.sd_disable_nsfw or args.disable_nsfw,
        # sd_cpu_textencoder=args.sd_cpu_textencoder,
        # sd_run_local=args.sd_run_local,
        # local_files_only=args.local_files_only,
        # cpu_offload=args.cpu_offload,
        # enable_xformers=args.sd_enable_xformers or args.enable_xformers,
        callback=diffuser_callback,
    )

    form = {
        "ldmSteps": 25,
        "ldmSampler": "plms",
        "zitsWireframe": True,
        "hdStrategy": "Crop",
        "hdStrategyCropMargin": 196,
        "hdStrategyCropTrigerSize": 800,
        "hdStrategyResizeLimit": 2048,
        "prompt": "",
        "negativePrompt": "",
        "croperX": 0,
        "croperY": 0,
        "croperHeight": 512,
        "croperWidth": 512,
        "useCroper": False,
        "sdMaskBlur": 5,
        "sdStrength": 0.75,
        "sdSteps": 50,
        "sdGuidanceScale": 7.5,
        "sdSampler": "pndm",
        "sdSeed": -1,
        "sdMatchHistograms": False,
        "sdScale": 1,
        "cv2Radius": 5,
        "cv2Flag": "INPAINT_NS",
        "paintByExampleSteps": 50,
        "paintByExampleGuidanceScale": 7.5,
        "paintByExampleSeed": -1,
        "paintByExampleMaskBlur": 5,
        "paintByExampleMatchHistograms": False,
        "p2pSteps": 50,
        "p2pImageGuidanceScale": 1.5,
        "p2pGuidanceScale": 7.5,
        "sizeLimit": 512,
    }

    image, alpha_channel, exif = load_img(original_image, return_exif=True)

    mask, _ = load_img(original_mask, gray=True)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    if image.shape[:2] != mask.shape[:2]:
        return (
            f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}",
            400,
        )

    original_shape = image.shape
    interpolation = cv2.INTER_CUBIC

    size_limit: Union[int, str] = form.get("sizeLimit", "1080")
    if size_limit == "Original":
        size_limit = max(image.shape)
    else:
        size_limit = int(size_limit)

    if "paintByExampleImage" in input:
        paint_by_example_example_image, _ = load_img(
            input["paintByExampleImage"].read()
        )
        paint_by_example_example_image = Image.fromarray(
            paint_by_example_example_image)
    else:
        paint_by_example_example_image = None

    config = Config(
        ldm_steps=form["ldmSteps"],
        ldm_sampler=form["ldmSampler"],
        hd_strategy=form["hdStrategy"],
        zits_wireframe=form["zitsWireframe"],
        hd_strategy_crop_margin=form["hdStrategyCropMargin"],
        hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
        hd_strategy_resize_limit=form["hdStrategyResizeLimit"],
        prompt=form["prompt"],
        negative_prompt=form["negativePrompt"],
        use_croper=form["useCroper"],
        croper_x=form["croperX"],
        croper_y=form["croperY"],
        croper_height=form["croperHeight"],
        croper_width=form["croperWidth"],
        sd_scale=form["sdScale"],
        sd_mask_blur=form["sdMaskBlur"],
        sd_strength=form["sdStrength"],
        sd_steps=form["sdSteps"],
        sd_guidance_scale=form["sdGuidanceScale"],
        sd_sampler=form["sdSampler"],
        sd_seed=form["sdSeed"],
        sd_match_histograms=form["sdMatchHistograms"],
        cv2_flag=form["cv2Flag"],
        cv2_radius=form["cv2Radius"],
        paint_by_example_steps=form["paintByExampleSteps"],
        paint_by_example_guidance_scale=form["paintByExampleGuidanceScale"],
        paint_by_example_mask_blur=form["paintByExampleMaskBlur"],
        paint_by_example_seed=form["paintByExampleSeed"],
        paint_by_example_match_histograms=form["paintByExampleMatchHistograms"],
        paint_by_example_example_image=paint_by_example_example_image,
        p2p_steps=form["p2pSteps"],
        p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
        p2p_guidance_scale=form["p2pGuidanceScale"],
    )

    if config.sd_seed == -1:
        config.sd_seed = random.randint(1, 999999999)
    if config.paint_by_example_seed == -1:
        config.paint_by_example_seed = random.randint(1, 999999999)

    logger.info(f"Origin image shape: {original_shape}")
    image = resize_max_size(image, size_limit=size_limit,
                            interpolation=interpolation)
    logger.info(f"Resized image shape: {image.shape}")

    mask = resize_max_size(mask, size_limit=size_limit,
                           interpolation=interpolation)

    start = time.time()
    try:
        res_np_img = model(image, mask, config)
    except RuntimeError as e:
        torch.cuda.empty_cache()
        if "CUDA out of memory. " in str(e):
            # NOTE: the string may change?
            return "CUDA out of memory", 500
        else:
            logger.exception(e)
            return "Internal Server Error", 500
    finally:
        logger.info(f"process time: {(time.time() - start) * 1000}ms")
        torch.cuda.empty_cache()

    res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != res_np_img.shape[:2]:
            alpha_channel = cv2.resize(
                alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
            )
        res_np_img = np.concatenate(
            (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
        )

    return Image.fromarray(res_np_img)
