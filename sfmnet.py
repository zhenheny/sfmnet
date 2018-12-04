from keras.layers import Dense,Deconv2D,Conv2D,BatchNormalization,Activation,Concatenate,Reshape,Conv1D,Permute
from keras.engine.topology import Layer

sess = tf.Session()

from keras import backend as K
K.set_session(sess)

def model_input(img_h,img_w,img_c):
    I_t0=tf.placeholder(tf.float32,(None,img_h,img_w,img_c),name='frame_t0')
    I_t1=tf.placeholder(tf.float32,(None,img_h,img_w,img_c),name='frame_t1')
    learning_rate=tf.placeholder(tf.float32)
    steering=tf.placeholder(tf.float32,(None,1),name='steering')
    
    return I_t0,I_t1,learning_rate,steering

with tf.name_scope("data_loading"):
    seed = random.randint(0, 2**31 - 1)
    dataset_dir = '/home/zhenheng/datasets/kitti/'
    num_scales = 4
    batch_size = 4
    num_source = 2
    # Load the list of training files into queues
    file_list = self.format_file_list(dataset_dir, 'train')
    image_paths_queue = tf.train.string_input_producer(
        file_list['image_file_list'], 
        seed=seed, 
        shuffle=False)
    cam_paths_queue = tf.train.string_input_producer(
        file_list['cam_file_list'], 
        seed=seed, 
        shuffle=False)

    # Load sequential images
    img_reader = tf.WholeFileReader()
    _, image_contents = img_reader.read(image_paths_queue)
    image_seq = tf.image.decode_jpeg(image_contents)
    image_seq = self.preprocess_image(image_seq)
    tgt_image, src_image_stack = \
            unpack_image_sequence(image_seq) # here the src_image_stack is a list, each element is B,H,W,3

    # Load camera intrinsics
    cam_reader = tf.TextLineReader()
    _, raw_cam_contents = cam_reader.read(cam_paths_queue)
    rec_def = []
    for i in range(9):
        rec_def.append([1.])
    raw_cam_vec = tf.decode_csv(raw_cam_contents, 
                                record_defaults=rec_def)
    raw_cam_vec = tf.stack(raw_cam_vec)
    raw_cam_mat = tf.reshape(raw_cam_vec, [3, 3])
    proj_cam2pix, proj_pix2cam = self.get_multi_scale_intrinsics(
        raw_cam_mat, num_scales)

    # Form training batches
    input_batch = tf.train.batch(src_image_stack + \
                    [tgt_image, proj_cam2pix, proj_pix2cam],
                     batch_size=batch_size,
                     num_threads=4,
                     capacity=64)
    src_image_stack = input_batch[:num_source]
    tgt_image, proj_cam2pix, proj_pix2cam = input_batch[num_source:]

    print ("tgt_image batch images shape:")
    print (tgt_image.get_shape().as_list())
    print ("src_image_stack[0] shape:")
    print (src_image_stack[0].get_shape().as_list())

img_h,img_w,img_c=128,384,3
I_t0,I_t1,learning_rate,steering=model_input(img_h,img_w,img_c)
assert I_t0.shape.as_list()==[None,128,384,3]
assert I_t1.shape.as_list()==[None,128,384,3]
assert steering.shape.as_list()==[None,1]

f_point_cloud_1,f_depth_output=structure_net(I_t0)
b_point_cloud_1,b_depth_output=structure_net(I_t1,reuse=True)

f_pix_pos,f_flow,f_point_cloud_2,f_motion_map=motion_net(I_t0,I_t1,f_point_cloud_1)
b_pix_pos,b_flow,b_point_cloud_2,b_motion_map=motion_net(I_t1,I_t0,b_point_cloud_1,reuse=True)

f_frame_loss=get_frame_loss()(I_t0,I_t1,f_pix_pos)
b_frame_loss=get_frame_loss()(I_t1,I_t0,b_pix_pos,reuse=True)

f_flow_sm_loss=get_smooth_loss(order=1)(f_flow,'flow')
b_flow_sm_loss=get_smooth_loss(order=1)(b_flow,'flow',reuse=True)

f_depth_sm_loss=get_smooth_loss(order=1)(f_depth_output,'depth')
b_depth_sm_loss=get_smooth_loss(order=1)(b_depth_output,'depth',reuse=True)

f_motion_sm_loss=get_smooth_loss(order=1)(f_motion_map,'motion')
b_motion_sm_loss=get_smooth_loss(order=1)(b_motion_map,'motion',reuse=True)

f_depth_loss=get_fb_depth_loss()(f_depth_output,b_depth_output,f_pix_pos,f_point_cloud_1)
b_depth_loss=get_fb_depth_loss()(b_depth_output,f_depth_output,b_pix_pos,b_point_cloud_1,reuse=True)

toal_loss=f_depth_loss+f_depth_sm_loss+f_motion_sm_loss+f_flow_sm_loss+f_frame_loss+\
   b_depth_loss+b_depth_sm_loss+b_motion_sm_loss+b_flow_sm_loss+b_frame_loss
train_op=tf.train.AdamOptimizer(learning_rate=0.0003,beta1=0.9).minimize(toal_loss)

tf.summary.scalar('toal_loss',toal_loss)
tf.summary.image('b_depth_output',b_depth_output,1)
tf.summary.image('f_depth_output',f_depth_output,1)

meger_summary=tf.summary.merge_all()

write=tf.summary.FileWriter('/tmp/sfm')
write.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver()
for step in range(10000):
    ckpt = tf.train.get_checkpoint_state('./checkpoint/')
    if (step==0) and ckpt and ckpt.model_checkpoint_path:
        print('load model')
        saver.restore(sess, ckpt.model_checkpoint_path)

    feed_dict={I_t0:test_frame0,I_t1: test_frame1}
    
    _ = sess.run([train_op],feed_dict=feed_dict)
#     

    if step%20==0:
        train_loss=sess.run(toal_loss,feed_dict=feed_dict)
        print(train_loss)
        
    if step%100==0:
        s=sess.run(meger_summary,feed_dict=feed_dict )
        write.add_summary(s,step)
        saver.save(sess, './checkpoint/'+'my-model', global_step=step)