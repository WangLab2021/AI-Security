# coding=utf-8
import os

from keras import backend as K
from keras.engine import Model
from keras.models import load_model
from keras.preprocessing import image

from baseline.evaluate import train_predict, test_predict, grid_result_eval, market_result_eval
from transfer.simple_rank_transfer import cross_entropy_loss


def train_pair_predict(pair_model_path, target_train_path, pid_path, score_path):
    model = load_model(pair_model_path)
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    train_predict(model, target_train_path, pid_path, score_path)

def extract_imgs(dir_path):
    imgs = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name:
            continue
        if 's' not in image_name:
            # grid
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1])
        elif 's' in image_name:
            # market
            arr = image_name.split('_')
            person = int(arr[0])
            camera = int(arr[1][1])
        else:
            continue
        image_path = os.path.join(dir_path, image_name)
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        imgs.append(x)
    return imgs


def tf_eucl_dist(inputs):
    x, y = inputs
    return K.square((x - y))

def avg_eucl_dist(inputs):
    x, y = inputs
    return K.mean(K.square((x - y)), axis=1)


def train_rank_predict(rank_model_path, target_train_path, pid_path, score_path):
    model = load_model(rank_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    train_predict(model, target_train_path, pid_path, score_path)


def test_rank_predict(rank_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    model = load_model(rank_model_path, custom_objects={'cross_entropy_loss': cross_entropy_loss})
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)


def grid_eval(source, transform_dir):
    target = 'grid'
    for i in range(10):
        test_pair_predict(source + '_pair_pretrain.h5',
                          transform_dir + 'cross%d' % i + '/probe', transform_dir + 'cross%d' % i + '/test',
                          source + '_' + target + '_pid.log', source + '_' + target + '_score.log')
        grid_result_eval(source + '_' + target + '_pid.log', 'gan.log')

def test_pair_predict(pair_model_path, target_probe_path, target_gallery_path, pid_path, score_path):
    # todo
    model = load_model(pair_model_path)
    # model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
                  outputs=[model.get_layer('resnet50').get_output_at(0)])
    # model = Model(inputs=[model.input], outputs=[model.get_layer('avg_pool').output])
    result_argsort = test_predict(model, target_probe_path, target_gallery_path, pid_path, score_path)
    return result_argsort

def market_eval(source, transform_dir):
    result_argsort = test_pair_predict(source + '_pair_pretrain.h5',
                          transform_dir + '/query', transform_dir + '/bounding_box_test',
                          'market_market_pid.log', 'market_market_score.log')
    return result_argsort


if __name__ == '__main__':
    result_argsort = market_eval('market', '../../dataset/Market-1501') #result_argsort为对每个probe来说，跟根据分数的gallery的下标
    market_result_eval(result_argsort,
                       TEST='../../dataset/Market-1501/bounding_box_test',
                       QUERY='../../dataset/Market-1501/query')


