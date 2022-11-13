from typing import Dict

import cv2
import logging

import kociemba
import numpy as np
import argparse
import os

logging.basicConfig(level=logging.DEBUG)


def quant_image(image, k):
    # Reshaping the image into a 2D array of pixels and 3 color values (BGR)
    float_image = np.float32(image).reshape(-1, 3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, kms_labels, center = cv2.kmeans(float_image, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    flattened_labels = kms_labels.flatten()
    final_img = center[flattened_labels]
    final_img = final_img.reshape(image.shape)
    return center, flattened_labels.reshape((1536, 2304, 1)), final_img


#  Up, Left, Front, Right, Back, and Down https://pypi.org/project/kociemba/
face_labels = ['U', 'R', 'F', 'D', 'L', 'B']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sub_folder", help="Image Folder (e.g. 'solve0'")
    args = parser.parse_args()
    logging.info(f"sub_folder: {args.sub_folder}")

    grid_mask = cv2.imread('grid_mask.png')
    dim = (768, 768)
    grid_mask = cv2.resize(grid_mask, dim, interpolation=cv2.INTER_LINEAR)

    lower_bound = np.array([200, 200, 200])
    upper_bound = np.array([255, 255, 255])
    mask_img = cv2.inRange(grid_mask, lower_bound, upper_bound)

    os.chdir(args.sub_folder)

    masked_images = []
    for face in face_labels:
        img_name = face + '.png'
        img_original = cv2.imread(img_name)
        img = cv2.resize(img_original, dim, interpolation=cv2.INTER_LINEAR)

        # Apply mask onto input image
        masked = cv2.bitwise_and(img, img, mask=mask_img)
        masked_images.append(masked)

    zero_one = np.hstack((masked_images[0], masked_images[1]))
    zero_one_two = np.hstack((zero_one, masked_images[2]))
    zero_one = None

    three_four = np.hstack((masked_images[3], masked_images[4]))
    three_four_five = np.hstack((three_four, masked_images[5]))
    three_four = None

    sticker_stack = np.vstack((zero_one_two, three_four_five))
    zero_one_two = None
    three_four_five = None

    # in the original mask here are the nine areas to count values
    bounded_boxes = [((87, 83), (235, 228)), ((318, 83), (464, 228)), ((548, 83), (694, 228)),
                     ((87, 299), (235, 445)), ((318, 299), (464, 445)), ((548, 299), (694, 445)),
                     ((87, 519), (235, 664)), ((318, 519), (464, 664)), ((548, 519), (694, 664))]

    #  Up, Left, Front, Right, Back, and Down https://pypi.org/project/kociemba/
    # the offset for the tiled images
    bounded_boxes_offsets = [
        (0, 0),
        (768, 0),
        (1536, 0),
        (0, 768),
        (768, 768),
        (1536, 768)
    ]

    cv2.imshow('sticker_stack', sticker_stack.copy())

    final_colours = [(0, 0, 0) for _ in range(9 * 6)]

    found_boxes = [False for _ in range(9 * 6)]

    regions_to_find = 7

    while True:
        color_counts, kmeans = analyze_colours(bounded_boxes, bounded_boxes_offsets,
                                               regions_to_find, sticker_stack, final_colours, found_boxes)

        cv2.imshow('sticker_stack_' + str(regions_to_find), sticker_stack.copy())
        cv2.imshow('kmeans_' + str(regions_to_find), kmeans.copy())

        new_regions_to_find, found_boxes = mask_found_colours(bounded_boxes, bounded_boxes_offsets, color_counts,
                                                              sticker_stack,
                                                              final_colours, found_boxes)

        logging.info(f"new_regions_to_find to find: {new_regions_to_find}")

        if new_regions_to_find <= 1:
            break

        if new_regions_to_find == regions_to_find:
            raise ValueError(f"Unable to improve finding stickers found only {7 - regions_to_find} colours")

        regions_to_find = new_regions_to_find

    colors_to_labels = compute_labels(final_colours)

    encoding = ""
    for colour in final_colours:
        encoding = encoding + colors_to_labels[colour]

    draw_final_stickers(bounded_boxes, bounded_boxes_offsets,
                        sticker_stack,
                        final_colours,
                        colors_to_labels)

    logging.info(f"final_colours: {final_colours}")
    logging.info(f"colors_to_labels: {colors_to_labels}")
    logging.info(f"encoding: {encoding}")
    cv2.imshow('final_stickers', sticker_stack.copy())

    cv2.waitKey()
    cv2.destroyAllWindows()

    solve = kociemba.solve(encoding)
    logging.info(f"solve:{solve}")


centres: dict[str, int] = {'U': 4 + (0 * 9), 'R': 4 + (1 * 9), 'F': 4 + (2 * 9), 'D': 4 + (3 * 9), 'L': 4 + (4 * 9),
                           'B': 4 + (5 * 9)}


def compute_labels(final_colours):
    colors_to_labels = {}
    u = final_colours[centres['U']]
    colors_to_labels[u] = 'U'
    r = final_colours[centres['R']]
    colors_to_labels[r] = 'R'
    f = final_colours[centres['F']]
    colors_to_labels[f] = 'F'
    d = final_colours[centres['D']]
    colors_to_labels[d] = 'D'
    l = final_colours[centres['L']]
    colors_to_labels[l] = 'L'
    b = final_colours[centres['B']]
    colors_to_labels[b] = 'B'
    return colors_to_labels


def mask_found_colours(bounded_boxes, bounded_boxes_offsets, color_counts, sticker_stack, final_labels, found_boxes):
    # where we see exactly 9 of a given colour that is a good detection. anything else is vague.
    label_found = {k: v == 9 for k, v in color_counts.items()}
    logging.info(f"label_found: {label_found}")
    absolute_box_index = 0
    for i, offset in enumerate(bounded_boxes_offsets):
        for box in bounded_boxes:
            if not found_boxes[absolute_box_index]:
                color = final_labels[absolute_box_index]
                found = label_found[color]
                found_boxes[absolute_box_index] = found
                if found:
                    ((x1, y1), (x2, y2)) = box
                    (dx, dy) = offset
                    x_low = x1 + dx
                    x_high = x2 + dx
                    y_low = y1 + dy
                    y_high = y2 + dy
                    start_point = (x_low, y_low)
                    end_point = (x_high, y_high)
                    cv2.rectangle(sticker_stack, start_point, end_point, (0, 0, 0), -1)
            absolute_box_index = absolute_box_index + 1
    regions_to_find = sum([0 if x else 1 for x in label_found.values()])
    logging.info(f"regions_to_find: {regions_to_find}, found_boxes: {found_boxes}")
    return regions_to_find, found_boxes


positions = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9',
             'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
             'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9']


def draw_final_stickers(bounded_boxes, bounded_boxes_offsets, sticker_stack, final_labels, colors_to_labels):
    absolute_box_index = 0
    for i, offset in enumerate(bounded_boxes_offsets):
        for box in bounded_boxes:
            color64 = final_labels[absolute_box_index]
            # convert data types int64 to int
            color = (int(color64[0]), int(color64[1]), int(color64[2]))
            ((x1, y1), (x2, y2)) = box
            (dx, dy) = offset
            x_low = x1 + dx
            x_high = x2 + dx
            y_low = y1 + dy
            y_high = y2 + dy
            start_point = (x_low, y_low)
            end_point = (x_high, y_high)
            cv2.rectangle(sticker_stack, start_point, end_point, color, -1)
            text = positions[absolute_box_index] + ' : ' + colors_to_labels[color64]
            cv2.putText(sticker_stack, text, start_point, fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1.5,
                        color=(250, 225, 100))
            absolute_box_index = absolute_box_index + 1
    return


colors = [
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 0, 255),
    (0, 0, 255)
]


def analyze_colours(bounded_boxes, bounded_boxes_offsets, regions_to_find, sticker_stack, final_colours, found_boxes):
    # Perform kmeans color segmentation
    centres, labels, kmeans = quant_image(sticker_stack, regions_to_find)
    color_counts = {(centres[i][0], centres[i][1], centres[i][2]): 0 for i in range(0, regions_to_find)}
    absolute_box_index = 0
    for i, offset in enumerate(bounded_boxes_offsets):
        for box in bounded_boxes:
            if not found_boxes[absolute_box_index]:
                ((x1, y1), (x2, y2)) = box
                (dx, dy) = offset
                x_low = x1 + dx
                x_high = x2 + dx
                y_low = y1 + dy
                y_high = y2 + dy
                label_counts = {}
                for x in range(x_low, x_high):
                    for y in range(y_low, y_high):
                        label = labels[y][x][0]
                        old = label_counts.get(label)
                        if not old:
                            label_counts[label] = 1
                        else:
                            label_counts[label] = old + 1

                max_label = max(label_counts, key=label_counts.get)
                centre = (centres[max_label][0], centres[max_label][1], centres[max_label][2])
                color_counts[centre] = color_counts[centre] + 1
                logging.info(
                    f" absolute_box_index: {absolute_box_index}, max_label: {max_label}, label_counts: {label_counts}")
                final_colours[absolute_box_index] = centre
            absolute_box_index = absolute_box_index + 1
    logging.info(f"color_counts: {color_counts}")
    return color_counts, kmeans


if __name__ == "__main__":
    main()
