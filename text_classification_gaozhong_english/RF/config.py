# coding=utf-8
class TrainingConfig(object):
    decay_step = 15000
    decay_rate = 0.99
    epoches = 50000
    evaluate_every = 500
    checkpoint_every = 500

class ModelConfig(object):
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]

    fc_layers = [1024, 1024]
    dropout_keep_prob = 0.9
    learning_rate = 0.0001

class Config(object):
#    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet = " etaoinsrhldc.umwgfypbv,k)(-102x\"j53?46:的7_9z/8;词q’一改你在处修下文写空格加!“中不错个有出学每多以语题误要作”该面并请意第用线增英上注行生划对内分及数容后同其除\\是删给人假计为求段和或间仅校横最小从可听右桌选句信如号节者两答只短缺时起他国开掉话字回交把符能使…点了余定单言共读名许漏参斜篇师活们已适课限项老%入结会据左连子当提贯之均自•允头所动料好根到材括李来现涉发地表填华于我∧大换将标高考此这﹩阅年成总包家得习细经‘至情主理法关事[实准示方应]三正*看息少感相完●封设报谈明合日书友边章与想原己就说约件外物过位做部议图电述&力体况白前心问美独教各因目试等解钟期班全建评介构观尾但化论由画真汇某秒确很照去天游长也绍按周重进历性讲达姓列无身业新称概通佳受车本助举社向邮次象些£展二简接何月￡望流没式影引都然她近工网机认演$公置母稿么朋种记整比常利遍卷例满赛星知任水道园乐保手规@持打父度故直→必】【告②①安市须越果◆则几品欢样征卡③系响放决程旅传什导而组气反帮难更亲＄广才平先里着被排食馆四取算务愿复调范世希代识爱员收环需片④"
    alphabet_size = len(alphabet)
    l0 = 1014
    batch_size = 1
    nums_classes = 236
    example_nums = 15846

    data_source = 'data/gaozhong/gaozhong_english/gaozhong_english_preprocessed.csv'
    label_source = 'data/gaozhong/gaozhong_english/labels_english.csv'

    training = TrainingConfig()

    model = ModelConfig()


config = Config()
