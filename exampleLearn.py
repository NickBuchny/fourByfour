### imports
import tensorflow as tf

index_max = 10
features = 5
width = 4
size = width*width

x_data = [[4, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 3],
	[0, 1, 1, 1, 1, 2, 1, 4, 1, 1, 1, 2, 2, 1, 1, 3]]

y_data = [[0, 4, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 3],
	[0, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2, 2, 4, 1, 3]]


# x_input = tf.constant( x_data , dtype=tf.int32 )
# y_ground = tf.constant( y_data , dtype=tf.int32 )

x_input = tf.placeholder( shape=[None,size] , dtype=tf.int32 )
y_ground = tf.placeholder( shape=[None,size] , dtype=tf.int32 )


embedding_vector = tf.Variable( tf.truncated_normal( [index_max, features] , dtype=tf.float32 ) )
embedding_vector_3_ranks = tf.reshape(embedding_vector, [1,index_max, features])

x_embedding = tf.nn.embedding_lookup(embedding_vector, x_input)

def conv_layer( x , layers_in=features , layers_out=features ):
  strides = [1, 1, 1, 1]
  w = tf.Variable( tf.truncated_normal([3,3,layers_in, layers_out], stddev=0.1, dtype=tf.float32) )
  b = tf.Variable( tf.constant(0.1, shape=[layers_out], dtype=tf.float32) )
  h = tf.nn.conv2d( x, w, strides=[1, 1, 1, 1], padding='SAME' ) + b
  return h # tf.nn.relu( h )

hidden = tf.reshape( x_embedding, [-1,width,width,features] )
hidden = tf.nn.relu( conv_layer( hidden ) )
hidden = tf.nn.relu( conv_layer( hidden ) )
hidden = tf.nn.relu( conv_layer( hidden ) )
y_output = tf.reshape( conv_layer( hidden ) , [-1,features] )


item_as_embedding = tf.tile( y_output , tf.constant([1,index_max]) )
item_as_embedding = tf.reshape( item_as_embedding , [-1,index_max,features] )

item_distance_to_embedding = tf.square( embedding_vector_3_ranks - item_as_embedding )
item_distance_to_embedding = tf.reduce_mean( item_distance_to_embedding , -1 )
item_distance_to_embedding = tf.reshape( item_distance_to_embedding, [-1,index_max] )

y_estimate = tf.arg_max( -1.0 * tf.reduce_mean( tf.square( tf.reshape( embedding_vector, [1,index_max, features]) - item_as_embedding ) , -1 ) , 1 )
mask = tf.reshape( tf.one_hot(y_ground,index_max, dtype=tf.float32) , [-1,index_max] )
error = tf.reduce_sum( mask * item_distance_to_embedding , -1 )
learn = tf.train.AdamOptimizer(0.001).minimize(error)


sess = tf.Session()
sess.run(tf.global_variables_initializer() )

for _ in range(20) :
    feed_dict = { x_input : x_data, y_ground : y_data }
    print sess.run(y_estimate,feed_dict).reshape([-1,size]) , sess.run(tf.reduce_sum(error,-1),feed_dict)
    for _ in range(20) :
      sess.run(learn,feed_dict)

