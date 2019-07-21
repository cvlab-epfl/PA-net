import numpy as np
from train.models import sample2d
from train import predict
import matplotlib.pyplot as plt
from skimage import io
import logging
from utils.utils_common import save_line, list_to_string, mkdir
logger = logging.getLogger(__name__)


class SynapseSlice:
    def __init__(self, x, y, y_hat, y_hat_heat_map, overlayed, jaccard_indx, slice_id, center):
        self.x = x
        self.y = y
        self.y_hat_heat_map = y_hat_heat_map
        self.y_hat = y_hat
        self.overlayed = overlayed
        self.jaccard_indx = jaccard_indx
        self.slice_id = slice_id
        self.center = center


class Synapse:

    def __init__(self, id):
        self.slices = []
        self.id = id

    def IoU(self):
        jaccards = [a.jaccard_indx for a in self.slices]
        return np.sum(jaccards, axis=0), len(jaccards)


def jaccard_index(image1, image2):
    _and = np.logical_and(image1, image2)
    _or = np.logical_or(image1, image2)

    if np.sum(_or) > 0:
        jaccard = np.sum(_and) / np.sum(_or)
    else:
        jaccard = 0  # be careful!

    return jaccard


def image_jaccard_index(x_slice, seed_slice, y_slice, unet_clsf, sampler, num_classes, seed_ids, slice_id, results):
    seeds_in_seed_slice = np.unique(seed_slice)
    seeds_in_seed_slice = np.delete(seeds_in_seed_slice, 0)

    for seed in seeds_in_seed_slice:

        if not str(seed) in results:
            results[str(seed)] = Synapse(seed)

        s = np.where(seed_ids[:, 1] == seed)[0]

        patch_o, patch_y_hat, patch_y, patch_x_small, patch_syn_small, center = predict.predict_patch(x_slice, seed_slice, y_slice, seed_id_vector, sampler, unet_clsf)

        jaccard_indices = np.zeros(num_classes)
        for j in range(num_classes):
            jaccard_indices[j] = jaccard_index(patch_y_hat == j, patch_y == j)

        overlayed = predict.overlay_slice(patch_x_small, patch_y_hat)

        patch_x_small = patch_x_small.patch[None]
        patch_syn_small = np.copy(patch_syn_small.patch[None])
        patch_syn_small[patch_syn_small != seed] = 0
        patch_syn_small[patch_syn_small == seed] = 1

        results[str(seed)].slices.append(SynapseSlice(np.concatenate((patch_x_small, patch_syn_small)), patch_y[0], patch_y_hat, patch_o[0], overlayed, jaccard_indices, slice_id, center))

    return results


def compute_loss(x, seed, y, results, sampler, num_classes, unet, idx):

    if seed is not None:
        slice_id = 1
        for (x_slice, seed_slice, y_slice) in zip(x, seed, y):
            print(str(slice_id) + ', ', end='', flush=True)
            image_jaccard_index(x_slice, seed_slice, y_slice, unet, sampler, num_classes, idx , slice_id, results)
            slice_id += 1
        print('')

        IoU_avg = np.zeros(num_classes)
        syn_count = 0
        for k in results:
            IoU, count = results[k].IoU()
            IoU_avg += IoU
            syn_count += count
        IoU_avg /= syn_count
        logger.info('   IoU: ' + list_to_string(IoU_avg))
    else:

        IoU_avg = np.zeros(num_classes)
        for i, (x_slice, y_slice) in enumerate(zip(x, y)):
            sample_patch = sample2d.SamplePatch(x_slice,y_slice, None, lambda x:x, sampler.in_patch_shape, sampler.out_patch_shape, unet.config.num_input_channels)

            # Get required patches
            patch_x, patch_y, patch_w = sample_patch.get_next_patch()

            # Predict
            patch_o = unet.forward(patch_x)
            patch_y_hat = np.argmax(patch_o[0], axis=0)

            centre = tuple([int(i / 2) for i in patch_y_hat.shape])
            slices = [slice(mid - int(np.floor(d/2)), mid + int(np.ceil(d/2))) for d, mid in zip(y_slice.shape, centre)]
            patch_y_hat= patch_y_hat[slices]


            results[i] = patch_y_hat #patch_o[:,:,slices[0],slices[1]].shape #, predict.blend(x_slice,patch_y_hat)

            IoU = np.zeros(num_classes)
            for j in range(num_classes):
                IoU[j] = jaccard_index(patch_y_hat == j, y_slice == j)

            IoU_avg += IoU
        IoU_avg /= len(x)
    return IoU_avg

def test(unet, all_data, sampler, save_path):

    # result parent dir
    save_path = save_path + '/IoUs'
    mkdir(save_path)

    all_results = {}
    for tag, d in all_data.items():
        logger.info(tag + ' set evaluation...')
        results = dict()
        loss = compute_loss(d.x, d.seed, d.y, results, sampler, unet.config.num_classes, unet, d.seed_ids)
        all_results[tag] = {'loss':loss, 'predictions':results}

        save_line(list_to_string(loss), save_path + '/' + tag + '_IoU.txt')

    return all_results


def save_predictions(iter, all_results, unet, save_path):

    results_for_log = dict()
    # result parent dir
    if unet.config.more_options.synapse_layer_mode=='synapse_dtf':
        save_path = save_path + '/doc_manual_label_synapses_dtf'
    elif unet.config.more_options.synapse_layer_mode=='center_dot_dtf':
        save_path = save_path + '/doc_manual_label_center_dot_dtf'
    mkdir(save_path)

    # result dir for the niter
    save_path = save_path + '/itr_' + str(iter)
    mkdir(save_path)

    for tag, results in all_results.items():
        mkdir(save_path + '/' + tag)
        if unet.config.more_options.dataset_name == 'EM':
            write_to_disk_em(results['predictions'], unet, save_path, tag, results_for_log)
        else:
            write_to_disk(results['predictions'], unet, save_path, tag, results_for_log)


    return results_for_log


def write_to_disk(results, unet, save_at, mode, results_for_log):
    save_at = save_at + '/' + mode
    for i, y_hat in (results.items()):
        y_hat = 255 * np.float32(y_hat) / unet.config.num_classes
        io.imsave(save_at + '/y_hat_' + str(i) + '.png', np.uint8(y_hat))


def write_to_disk_em(synapse_list, unet, save_at, mode, results_for_log):
    gap = np.ones((388, 20))


    cmap = plt.get_cmap('jet')

    jc_sum_all = np.zeros(unet.config.num_classes)
    jc_count_all = 0

    save_at = save_at + '/' + mode
    for syn in synapse_list:
        syn_path = save_at + '/' + syn

        mkdir(syn_path)
        mkdir(syn_path + '/y_hat')
        mkdir(syn_path + '/y_hat_overlayed')

        jc_sum, count = synapse_list[syn].IoU()

        jc_sum_all += jc_sum
        jc_count_all += count
        save_line(list_to_string(jc_sum/count), syn_path + '/jaccard_average_synapse.txt')

        for slice in synapse_list[syn].slices:


            y_hat = 255 * slice.y_hat / unet.config.num_classes
            io.imsave(syn_path + '/y_hat/y_hat_' + str(slice.slice_id) + '.png', np.uint8(y_hat))
            io.imsave(syn_path + '/y_hat_overlayed/y_hat_overlayed_' + str(slice.slice_id) + '.png', slice.overlayed)

            save_line(list_to_string(slice.jaccard_indx), syn_path + '/jaccard_slice.txt')
            save_line(str(slice.slice_id) + ' : ' + list_to_string(slice.center), syn_path + '/centers.txt')

    jc_avg = jc_sum_all/jc_count_all
    for i in range(unet.config.num_classes):
        results_for_log[mode + '_jaccard_index_' + str(i)] =  jc_avg[i]

    save_line(list_to_string(jc_avg), save_at + '/jaccard_average_all.txt')

def write_to_disk_em_old(synapse_list, unet, save_at, mode, results_for_log):
    gap = np.ones((388, 20))


    cmap = plt.get_cmap('jet')

    jc_sum_all = np.zeros(unet.config.num_classes)
    jc_count_all = 0

    save_at = save_at + '/' + mode
    for syn in synapse_list:
        syn_path = save_at + '/' + syn

        mkdir(syn_path)
        mkdir(syn_path + '/x')
        mkdir(syn_path + '/y')
        mkdir(syn_path + '/y_hat_heat_map')
        mkdir(syn_path + '/y_hat')
        mkdir(syn_path + '/y_hat_overlayed')

        jc_sum, count = synapse_list[syn].IoU()

        jc_sum_all += jc_sum
        jc_count_all += count
        save_line(list_to_string(jc_sum/count), syn_path + '/jaccard_average_synapse.txt')

        for slice in synapse_list[syn].slices:
            x = slice.x
            io.imsave(syn_path + '/x/x_0' + str(slice.slice_id) + '.png', np.uint8(255 * x[0]))
            io.imsave(syn_path + '/x/x_1' + str(slice.slice_id) + '.png', np.uint8(255 * x[1]))

            y = 255 * slice.y / unet.config.num_classes
            io.imsave(syn_path + '/y/y_' + str(slice.slice_id) + '.png', np.uint8(y))

            y_hat = 255 * slice.y_hat / unet.config.num_classes
            io.imsave(syn_path + '/y_hat/y_hat_' + str(slice.slice_id) + '.png', np.uint8(y_hat))

            heat_maps = slice.y_hat_heat_map

            color_heat_maps = [np.delete(cmap(map), 3, 2) for map in heat_maps]
            #color_heat_maps = np.concatenate( (color_heat_maps[0], gap3, color_heat_maps[1], gap3, color_heat_maps[2]), axis=1)

            color_heat_maps = np.concatenate( [np.concatenate((map, np.ones((map.shape[0], 20, 3))), axis=1) for map in color_heat_maps ], axis=1)
            color_heat_maps = np.uint8(255 * color_heat_maps)
            io.imsave(syn_path + '/y_hat_heat_map/y_hat_heat_map_' + str(slice.slice_id) + '.png', color_heat_maps)

            io.imsave(syn_path + '/y_hat_overlayed/y_hat_overlayed_' + str(slice.slice_id) + '.png', slice.overlayed)

            save_line(list_to_string(slice.jaccard_indx), syn_path + '/jaccard_slice.txt')
            save_line(str(slice.slice_id) + ' : ' + list_to_string(slice.center), syn_path + '/centers.txt')

    jc_avg = jc_sum_all/jc_count_all
    for i in range(unet.config.num_classes):
        results_for_log[mode + '_jaccard_index_' + str(i)] =  jc_avg[i]

    save_line(list_to_string(jc_avg), save_at + '/jaccard_average_all.txt')