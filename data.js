const logging = require("logging");
const math = require("mathjs");
const torch = require("torch");
const nn = require("torch.nn");
const torchaudio = require("torchaudio");
const Image = require("PIL");
const pv_transforms = require("pytorchvideo.transforms");
const ConstantClipsPerVideoSampler = require("pytorchvideo.data.clip_sampling.ConstantClipsPerVideoSampler");
const EncodedVideo = require("pytorchvideo.data.encoded_video");
const transforms = require("torchvision.transforms");
const NormalizeVideo = require("torchvision.transforms._transforms_video.NormalizeVideo");
const SimpleTokenizer = require("./models/multimodal_preprocessors/SimpleTokenizer");

const DEFAULT_AUDIO_FRAME_SHIFT_MS = 10;

const BPE_PATH = "bpe/bpe_simple_vocab_16e6.txt.gz";

function waveform2melspec(waveform, sample_rate, num_mel_bins, target_length) {
  waveform -= waveform.mean();
  const fbank = torchaudio.compliance.kaldi.fbank(
    waveform,
    true,
    sample_rate,
    false,
    "hanning",
    num_mel_bins,
    0.0,
    25,
    DEFAULT_AUDIO_FRAME_SHIFT_MS
  );
  const fbankTranspose = torch.transpose(fbank, 0, 1);
  const n_frames = fbankTranspose.size(1);
  const p = target_length - n_frames;
  if (Math.abs(p) / n_frames > 0.2) {
    logging.warning(
      `Large gap between audio n_frames(${n_frames}) and target_length (${target_length}). Is the audio_target_length setting correct?`
    );
  }
  let fbankPadded;
  if (p > 0) {
    fbankPadded = torch.nn.functional.pad(
      fbankTranspose,
      [0, p],
      "constant",
      0
    );
  } else if (p < 0) {
    fbankPadded = fbankTranspose.narrow(1, 0, target_length);
  } else {
    fbankPadded = fbankTranspose;
  }
  const fbankUnsqueezed = fbankPadded.unsqueeze(0);
  return fbankUnsqueezed;
}

function get_clip_timepoints(clip_sampler, duration) {
  const all_clips_timepoints = [];
  let is_last_clip = false;
  let end = 0.0;
  while (!is_last_clip) {
    let start, end, _, __, is_last_clip;
    [start, end, _, __, is_last_clip] = clip_sampler(end, duration, null);
    all_clips_timepoints.push([start, end]);
  }
  return all_clips_timepoints;
}

function load_and_transform_vision_data(image_paths, device) {
  if (image_paths === null) {
    return null;
  }

  const image_outputs = [];
  for (let i = 0; i < image_paths.length; i++) {
    const data_transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]
      ),
    ]);
    const image = Image.open(image_paths[i]).convert("RGB");
    const imageToTensor = data_transform(image).to(device);
    image_outputs.push(imageToTensor);
  }
  return torch.stack(image_outputs, 0);
}

function loadAndTransformText(text, device) {
  if (text === null) {
    return null;
  }
  const tokenizer = new SimpleTokenizer(BPE_PATH);
  const tokens = text.map((t) => tokenizer(t).unsqueeze(0).to(device));
  const concatenatedTokens = torch.cat(tokens, 0);
  return concatenatedTokens;
}

function loadAndTransformAudioData(
  audioPaths,
  device,
  numMelBins = 128,
  targetLength = 204,
  sampleRate = 16000,
  clipDuration = 2,
  clipsPerVideo = 3,
  mean = -4.268,
  std = 9.138
) {
  if (audioPaths === null) {
    return null;
  }

  const audioOutputs = [];
  const clipSampler = new ConstantClipsPerVideoSampler(
    clipDuration,
    clipsPerVideo
  );

  for (const audioPath of audioPaths) {
    const waveform = torchaudio.load(audioPath);
    const sr = waveform[1];
    if (sampleRate !== sr) {
      waveform = torchaudio.functional.resample(waveform, sr, sampleRate);
    }
    const allClipsTimepoints = getClipTimepoints(
      clipSampler,
      waveform.size(1) / sampleRate
    );
    const allClips = [];
    for (const clipTimepoints of allClipsTimepoints) {
      const waveformClip = waveform.slice(
        null,
        null,
        Math.floor(clipTimepoints[0] * sampleRate),
        Math.floor(clipTimepoints[1] * sampleRate)
      );
      const waveformMelspec = waveform2melspec(
        waveformClip,
        sampleRate,
        numMelBins,
        targetLength
      );
      allClips.push(waveformMelspec);
    }

    const normalize = new transforms.Normalize(mean, std);
    const normalizedClips = allClips.map((ac) => normalize(ac).to(device));

    const stackedClips = torch.stack(normalizedClips, 0);
    audioOutputs.push(stackedClips);
  }

  return torch.stack(audioOutputs, 0);
}

function cropBoxes(boxes, xOffset, yOffset) {
  const croppedBoxes = boxes.clone();
  croppedBoxes.select([0, 2]).sub_(xOffset);
  croppedBoxes.select([1, 3]).sub_(yOffset);

  return croppedBoxes;
}

function uniformCrop(images, size, spatialIdx, boxes = null, scaleSize = null) {
  if (![0, 1, 2].includes(spatialIdx)) {
    throw new Error("Invalid spatial index");
  }

  const ndim = images.shape.length;
  if (ndim === 3) {
    images.unsqueeze_(0);
  }
  const height = images.shape[2];
  const width = images.shape[3];

  if (scaleSize !== null) {
    if (width <= height) {
      width = scaleSize;
      height = Math.floor((height / width) * scaleSize);
    } else {
      width = Math.floor((width / height) * scaleSize);
      height = scaleSize;
    }
    images = torch.nn.functional.interpolate(images, [height, width], {
      mode: "bilinear",
      align_corners: false,
    });
  }

  let yOffset = Math.ceil((height - size) / 2);
  let xOffset = Math.ceil((width - size) / 2);

  if (height > width) {
    if (spatialIdx === 0) {
      yOffset = 0;
    } else if (spatialIdx === 2) {
      yOffset = height - size;
    }
  } else {
    if (spatialIdx === 0) {
      xOffset = 0;
    } else if (spatialIdx === 2) {
      xOffset = width - size;
    }
  }
  const cropped = images.slice(
    null,
    null,
    yOffset,
    yOffset + size,
    xOffset,
    xOffset + size
  );
  const croppedBoxes =
    boxes !== null ? cropBoxes(boxes, xOffset, yOffset) : null;
  if (ndim === 3) {
    cropped.squeeze_(0);
  }
  return [cropped, croppedBoxes];
}

class SpatialCrop extends nn.Module {
  constructor(cropSize = 224, numCrops = 3) {
    super();
    this.cropSize = cropSize;
    if (numCrops === 3) {
      this.cropsToExt = [0, 1, 2];
      this.flippedCropsToExt = [];
    } else if (numCrops === 1) {
      this.cropsToExt = [1];
      this.flippedCropsToExt = [];
    } else {
      throw new Error("Nothing else supported yet");
    }
  }

  forward(videos) {
    if (!Array.isArray(videos)) {
      throw new Error("Must be a list of videos after temporal crops");
    }
    if (!videos.every((video) => video.ndim === 4)) {
      throw new Error("Must be (C,T,H,W)");
    }
    const res = [];
    for (const video of videos) {
      for (const spatialIdx of this.cropsToExt) {
        res.push(uniformCrop(video, this.cropSize, spatialIdx)[0]);
      }
      if (!this.flippedCropsToExt.length) {
        continue;
      }
      const flippedVideo = transforms.functional.hflip(video);
      for (const spatialIdx of this.flippedCropsToExt) {
        res.push(uniformCrop(flippedVideo, this.cropSize, spatialIdx)[0]);
      }
    }
    return res;
  }
}

function load_and_transform_video_data(
  video_paths,
  device,
  clip_duration = 2,
  clips_per_video = 5,
  sample_rate = 16000
) {
  if (video_paths === null) {
    return null;
  }

  const video_outputs = [];
  const video_transform = transforms.Compose([
    pv_transforms.ShortSideScale(224),
    NormalizeVideo({
      mean: [0.48145466, 0.4578275, 0.40821073],
      std: [0.26862954, 0.26130258, 0.27577711],
    }),
  ]);

  const clip_sampler = ConstantClipsPerVideoSampler({
    clip_duration: clip_duration,
    clips_per_video: clips_per_video,
  });
  const frame_sampler = pv_transforms.UniformTemporalSubsample({
    num_samples: clip_duration,
  });

  for (const video_path of video_paths) {
    const video = EncodedVideo.from_path(video_path, {
      decoder: "decord",
      decode_audio: false,
      sample_rate: sample_rate,
    });

    const all_clips_timepoints = get_clip_timepoints(
      clip_sampler,
      video.duration
    );

    const all_video = [];
    for (const clip_timepoints of all_clips_timepoints) {
      // Read the clip, get frames
      const clip = video.get_clip(clip_timepoints[0], clip_timepoints[1]);
      if (clip === null) {
        throw new Error("No clip found");
      }
      let video_clip = frame_sampler(clip["video"]);
      video_clip = video_clip.div(255.0); // since this is float, need 0-1

      all_video.push(video_clip);
    }

    const transformed_video = all_video.map((clip) => video_transform(clip));
    const cropped_video = new SpatialCrop(224, (num_crops = 3)).forward(
      transformed_video
    );

    const stacked_video = torch.stack(cropped_video, 0);
    video_outputs.push(stacked_video);
  }

  return torch.stack(video_outputs, 0).to(device);
}
