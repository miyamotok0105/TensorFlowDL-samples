import tensorflow as tf

#tf.app.flags.FLAGSでパラメータ付与できる
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('data_num', 100, """データ数""")
tf.app.flags.DEFINE_string('img_path', './Baskerville-01.png', """画像ファイルパス""")

#mainが実行される
def main(argv):
    print("main!")
    print(FLAGS.data_num, FLAGS.img_path)

if __name__ == '__main__':
    #tf.app.runでmainを実行する
    tf.app.run()

