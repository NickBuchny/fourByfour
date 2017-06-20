### imports
import tensorflow as tf

image_depth = 5
image_width = 4
image_size = image_width*image_width
move_depth = 2

features = 7

x_data = [[4, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 3],
  [4, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 3],
  [0, 1, 1, 1, 1, 2, 1, 4, 1, 1, 1, 2, 2, 1, 1, 3]]

m_data = [0,1,0]

y_data = [[0, 4, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 3],
  [0, 1, 1, 1, 4, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 3],
  [0, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2, 2, 4, 1, 3]]

x_input = tf.placeholder( shape=[None,image_size] , dtype=tf.int32 )
m_input = tf.placeholder( shape=[None] , dtype=tf.int32 )
y_ground = tf.placeholder( shape=[None,image_size] , dtype=tf.int32 )

x_one_hot = tf.one_hot( x_input , image_depth )
m_one_hot = tf.one_hot( m_input , move_depth )
y_ground_hot = tf.one_hot( y_ground , image_depth )

def m_effector( m , layers_in=move_depth , layers_out=features ):
  w = tf.Variable( tf.truncated_normal( [layers_in, layers_out], stddev=0.1, dtype=tf.float32) )
  b = tf.Variable( tf.constant(0.1, shape=[layers_out], dtype=tf.float32) )
  h = tf.reshape( tf.matmul(m,w) + b , [-1,1,1,features] )
  return h

def conv_layer( x , layers_in=features , layers_out=features ):
  strides = [1, 1, 1, 1]
  w = tf.Variable( tf.truncated_normal([3,3,layers_in, layers_out], stddev=0.1, dtype=tf.float32) )
  b = tf.Variable( tf.constant(0.1, shape=[layers_out], dtype=tf.float32) )
  h = tf.nn.conv2d( x, w, strides=[1, 1, 1, 1], padding='SAME' ) + b
  return h

hidden = tf.reshape( x_one_hot, [-1, image_width, image_width, image_depth] )
hidden = tf.nn.relu( conv_layer( hidden , image_depth , features ) )
hidden = hidden + tf.nn.relu( m_effector( m_one_hot ) )
hidden = tf.nn.relu( conv_layer( hidden ) )
hidden = hidden + tf.nn.relu( m_effector( m_one_hot ) )
hidden = tf.nn.relu( conv_layer( hidden ) )
hidden = hidden + tf.nn.relu( m_effector( m_one_hot ) )
y_out  = tf.reshape( conv_layer( hidden , features, image_depth) , [-1, image_size, image_depth] )

y_estimate = tf.arg_max( y_out , -1 )
error = tf.reduce_sum( tf.square( y_ground_hot - y_out ) , -1 )
learn = tf.train.AdamOptimizer(0.001).minimize(error)

sess = tf.Session()
sess.run(tf.global_variables_initializer() )

for _ in range(20) :
    feed_dict = { x_input : x_data, y_ground : y_data, m_input : m_data }
    print sess.run(y_estimate,feed_dict).reshape([-1,image_size]) , sess.run(tf.reduce_sum(error,-1),feed_dict)
    for _ in range(20) :
      sess.run(learn,feed_dict)

