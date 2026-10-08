"""Create compact, labelled derivatives; source simulator recordings stay untouched."""
from pathlib import Path
import hashlib
import json
import sys

from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path('/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder')
SOURCE = ROOT.parent / 'lehome-fold-repro'
OUT = ROOT / 'docs/gifs'
CAMPAIGN = 'campaigns/20260920-media-root-fix/outputs/adapt_s0_Top_Short_Seen_0_pose2'
FONT = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'
CASES = [
    dict(name='folding-policy-success',
         clip='results/rollout_Pant_Short_Seen_0_repro_ep509_policy_success.gif',
         evidence='results/rollout_Pant_Short_Seen_0_repro_ep509.json',
         camera='top', crop=(640, 0, 1280, 480), width=512,
         success=True, pose=509, variant='historical_smolvla_raster_ft_full',
         title='CHECKER SUCCESS | Historical policy',
         detail='Short pants | Top camera | Learned actions',
         timing='Adjusted playback | Selected development episode'),
    dict(name='folding-policy-failure',
         clip='results/rollout_Pant_Short_Seen_0_repro_ep503_policy_failure.gif',
         evidence='results/rollout_Pant_Short_Seen_0_repro_ep503.json',
         camera='top', crop=(640, 0, 1280, 480), width=512,
         success=False, pose=503, variant='historical_smolvla_raster_ft_full',
         title='CHECKER FAILURE | Historical policy',
         detail='Short pants | Top camera | Learned actions',
         timing='Adjusted playback | Selected development episode'),
    dict(name='folding-adapted-failure',
         clip=f'{CAMPAIGN}/rollout_policy_failure_top.gif',
         evidence=f'{CAMPAIGN}/rollout.json',
         camera='top', crop=None, width=512,
         success=False, pose=2, variant='adapt_s0',
         title='CHECKER FAILURE | Adapted policy',
         detail='Short-sleeve top | Seed 0 | Top camera',
         timing='0.5x simulation playback + end hold | 600 actions'),
    dict(name='folding-adapted-left-wrist',
         clip=f'{CAMPAIGN}/rollout_policy_failure_left_wrist.gif',
         evidence=f'{CAMPAIGN}/rollout.json',
         camera='left_wrist', crop=None, width=384,
         success=False, pose=2, variant='adapt_s0',
         title='CHECKER FAILURE | Left wrist',
         detail='Adapted policy, seed 0 | Short-sleeve top',
         timing='0.5x simulation playback + end hold'),
    dict(name='folding-adapted-right-wrist',
         clip=f'{CAMPAIGN}/rollout_policy_failure_right_wrist.gif',
         evidence=f'{CAMPAIGN}/rollout.json',
         camera='right_wrist', crop=None, width=384,
         success=False, pose=2, variant='adapt_s0',
         title='CHECKER FAILURE | Right wrist',
         detail='Adapted policy, seed 0 | Short-sleeve top',
         timing='0.5x simulation playback + end hold'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_font(text, size, width):
    while size >= 9:
        font = ImageFont.truetype(FONT, size)
        if font.getlength(text) <= width - 24:
            return font
        size -= 1
    raise ValueError(f'Caption is too wide: {text}')


def main():
    for case in CASES:
        output = OUT / (case['name'] + '.gif')
        sidecar = output.with_suffix('.json')
        poster = output.with_suffix('.png')
        if any(p.exists() for p in (output, sidecar, poster)):
            raise FileExistsError(f'Refusing to overwrite {case["name"]}')
        source_clip = SOURCE / case['clip']
        evidence = SOURCE / case['evidence']
        result = json.loads(evidence.read_text())
        if result['mode'] != 'policy' or result['success'] is not case['success']:
            raise ValueError('Recorded policy verdict does not match caption')
        modern = case['variant'] == 'adapt_s0'
        if modern:
            if not result['episode_complete'] or result['completed_steps'] != 600:
                raise ValueError('Expected a completed 600-action rollout')
            if result['terminal_success'] or result['n_rendered'] != 601:
                raise ValueError('Unexpected adapted episode outcome/render accounting')
            if Path(result['gif_views'][case['camera']]) != source_clip:
                raise ValueError('Camera identity disagrees with the measured result')

        width = case['width']
        height = width * 3 // 4
        fonts = [fit_font(case[key], size, width)
                 for key, size in [('title', 18), ('detail', 13), ('timing', 12)]]
        frames, source_durations, durations = [], [], []
        with Image.open(source_clip) as original:
            original_size = list(original.size)
            source_frame_count = original.n_frames
            for frame in ImageSequence.Iterator(original):
                source_duration = frame.info.get('duration')
                source_durations.append(source_duration)
                image = frame.convert('RGB')
                if case['crop']:
                    if image.size != (1920, 480):
                        raise ValueError('Historical source no longer has the documented left/top/right layout')
                    image = image.crop(case['crop'])
                image = image.resize((width, height), Image.Resampling.LANCZOS)
                labelled = Image.new('RGB', (width, height + 72), '#12171d')
                labelled.paste(image, (0, 0))
                draw = ImageDraw.Draw(labelled)
                accent = '#59ddaf' if case['success'] else '#ffb39e'
                for key, y, font, color in zip(('title', 'detail', 'timing'),
                                              (height + 8, height + 34, height + 52),
                                              fonts, (accent, '#f1f5f9', '#bac8d6')):
                    draw.text((12, y), case[key], font=font, fill=color)
                frames.append(labelled)
                if modern:
                    if source_duration is None or source_duration <= 0:
                        raise ValueError('Modern clip lost its simulation-based timestamps')
                    durations.append(source_duration * 2)
                else:
                    durations.append(100)
        durations[-1] += 1000

        # One palette avoids frame-to-frame color changes; repeated identical
        # frames may merge in the GIF while retaining their total duration.
        sample = Image.new('RGB', (96, 72 * len(frames)))
        for i, frame in enumerate(frames):
            sample.paste(frame.resize((96, 72)), (0, 72 * i))
        palette = sample.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
        quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        quantized[0].save(output, save_all=True, append_images=quantized[1:],
                          duration=durations, loop=0, optimize=True, disposal=1)
        poster_index = len(frames) // 2
        frames[poster_index].save(poster, optimize=True)

        with Image.open(output) as encoded:
            output_size = list(encoded.size)
            encoded_frames = encoded.n_frames
            encoded_duration_ms = sum(frame.info.get('duration', 0)
                                      for frame in ImageSequence.Iterator(encoded))
        if encoded_duration_ms != sum(durations):
            raise ValueError('Encoded GIF timing does not match the documented playback')
        if output.stat().st_size > 3_000_000:
            raise ValueError(f'{output.name} exceeds the three-megabyte publication budget')
        record = {
            'schema_version': 1,
            'source_repository': 'https://github.com/joses2017smjh/IsaacSimFolding',
            'source_clip': case['clip'], 'source_sha256': sha(source_clip),
            'evidence': case['evidence'], 'evidence_sha256': sha(evidence),
            'source_scope': 'Source paths are relative to the IsaacSimFolding project; original artifacts are not bundled here.',
            'source_size': original_size, 'source_frame_count': source_frame_count,
            'source_camera': case['camera'],
            'source_triptych_order': ['left_wrist', 'top', 'right_wrist'] if case['crop'] else None,
            'crop_box_xyxy': list(case['crop']) if case['crop'] else None,
            'mode': 'policy', 'replayed_demonstration_actions': False,
            'policy_variant': case['variant'], 'garment': result['garment'],
            'pose_source_episode': case['pose'],
            'reported_actions': result['steps'],
            'completed_actions_explicitly_recorded': result.get('completed_steps'),
            'checker': 'LeHome success_checker_garment_fold',
            'success': result['success'], 'terminal_success': result.get('terminal_success'),
            'first_success_step': result.get('first_success_step'),
            'success_scope': ('The checker never passed during the fixed 600-action budget and also failed at the terminal state.' if modern
                              else 'Historical success is latched when the checker passes at any step; terminal success and first-success step were not recorded.'),
            'benchmark_scope': 'One selected development rollout, not a success-rate estimate or a comparison between policy generations.',
            'renderer': 'Storm rasterized RGB',
            'simulation_seconds': result.get('simulation_seconds'),
            'cloth_max_particle_displacement_m': result.get('cloth_motion', {}).get('max_particle_displacement'),
            'playback': {
                'description': case['timing'],
                'simulation_speed_factor': 0.5 if modern else None,
                'source_timing': ('Simulation-based GIF timestamps, 60/70 milliseconds between captured frames.' if modern
                                  else 'Source GIF contains missing or zero frame delays; physical-time playback cannot be recovered from this artifact.'),
                'historical_assigned_frame_ms': None if modern else 100,
                'extra_final_hold_ms': 1000,
                'output_duration_seconds': encoded_duration_ms / 1000,
                'all_source_frames_processed': True,
                'identical_frames_may_be_merged': True,
            },
            'transformations': ['Native simulator footage; no generated or interpolated action frames.',
                                'Lanczos spatial resizing; static caption footer; fixed 96-color GIF palette.',
                                'Historical top camera extracted from the middle third.' if case['crop'] else 'Entire native camera frame retained.'],
            'caption': case['title'], 'scope_caption': case['detail'],
            'output': str(output.relative_to(ROOT)), 'output_sha256': sha(output),
            'output_bytes': output.stat().st_size, 'output_size': output_size,
            'output_frame_count': encoded_frames,
            'poster': str(poster.relative_to(ROOT)), 'poster_sha256': sha(poster),
            'poster_source_frame_index': poster_index,
        }
        sidecar.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'name': case['name'], 'bytes': output.stat().st_size,
                          'frames': encoded_frames, 'duration_s': encoded_duration_ms / 1000}), flush=True)


if __name__ == '__main__':
    main()
