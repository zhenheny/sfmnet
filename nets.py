def conv(h_0,filters,kernel_size,strides):
        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.0001)
    
        h1=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer=kernel_initializer)(h_0)
        h1_bn=BatchNormalization()(h1,training=True)
        h1_o=Activation('relu')(h1_bn)
        return h1_o
def deconv(h_0,filters,kernel_size,strides):
        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.0001)
    
        h1=Deconv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer=kernel_initializer)(h_0)
        h1_bn=BatchNormalization()(h1,training=True)
        h1_o=Activation('relu')(h1_bn)
        return h1_o

def conv_deconv_net(frame):

    ###to deconv
    h10_o=conv(frame,filters=32,kernel_size=3,strides=1)
    h11_o=conv(h10_o,filters=64,kernel_size=3,strides=2)
    ###to deconv
    h20_o=conv(h11_o,filters=64,kernel_size=3,strides=1)
    h21_o=conv(h20_o,filters=128,kernel_size=3,strides=2)

    ###to deconv
    h30_o=conv(h21_o,filters=128,kernel_size=3,strides=1)
    h31_o=conv(h30_o,filters=256,kernel_size=3,strides=2)
    ###to deconv
    h40_o=conv(h31_o,filters=256,kernel_size=3,strides=1)
    h41_o=conv(h40_o,filters=512,kernel_size=3,strides=2)
    ###to deconv
    h50_o=conv(h41_o,filters=512,kernel_size=3,strides=1)
    h51_o=conv(h50_o,filters=1024,kernel_size=3,strides=2)
    #embeding layer
    embeding=conv(h51_o,filters=1024,kernel_size=3,strides=1)


    #deconv
    d5=deconv(embeding,filters=512,kernel_size=3,strides=2)
    d4_i=Concatenate(axis=-1)([d5,h50_o])
    d4=deconv(d4_i,filters=256,kernel_size=3,strides=2)
    d3_i=Concatenate(axis=-1)([d4,h40_o])
    d3=deconv(d3_i,filters=128,kernel_size=3,strides=2)
    d2_i=Concatenate(axis=-1)([d3,h30_o])
    d2=deconv(d2_i,filters=64,kernel_size=3,strides=2)
    d1_i=Concatenate(axis=-1)([d2,h20_o])
    out=deconv(d1_i,filters=32,kernel_size=3,strides=2)
    return out,embeding

def clip_relu(x):

    x=tf.clip_by_value(x, 1, 100)
    return x

def depth_net(frame):

    top,_=conv_deconv_net(frame)
    top=Conv2D(filters=1,kernel_size=1,strides=1,padding='same',kernel_initializer=keras.initializers.glorot_normal())(top)
    depth=Activation(clip_relu)(top)
    return depth

def structure_net(input_frame,reuse=False):
    with tf.variable_scope('structure_net',reuse=reuse):
        depth_output=depth_net(input_frame)
        point_cloud_output=Cloud_transformer()(depth_output)
        return point_cloud_output,depth_output

def sin_relu(x):
    x=tf.clip_by_value(x, -1., 1.)
    
    return x

def param_net(frame_t0,frame_t1,k_obj=4,):
    init=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.0001)
    frame_pair=tf.concat([frame_t0,frame_t1],-1)
    top,embed=conv_deconv_net(frame_pair)
    mask=Conv2D(filters=k_obj,kernel_size=1,strides=1,padding='same',kernel_initializer=init)(top)
    
    embed=Dense(512,kernel_initializer=init)(embed)
    embed=Dense(512,kernel_initializer=init)(embed)
    embed=Reshape([-1])(embed)

    cam_t_=Dense(3,kernel_initializer=init)(embed)
    cam_t=Activation('relu')(cam_t_)
    cam_p=Dense(600,kernel_initializer=init)(embed)
    cam_p=Activation('relu')(cam_p)
    cam_r=Dense(3,kernel_initializer=init)(embed)
    cam_r=Activation(sin_relu)(cam_r)
    
    obj_mask= Activation('sigmoid')(mask) 
    obj_t=Activation('relu')(Dense(3*k_obj,kernel_initializer=init)(embed))
    obj_t=tf.reshape(obj_t,(-1,k_obj,3))
    obj_p=Activation('relu')(Dense(600*k_obj,kernel_initializer=init)(embed))
    obj_p=tf.reshape(obj_p,(-1,k_obj,600))
    
    obj_r=Activation(sin_relu)(Dense(3*k_obj,kernel_initializer=init)(embed))
    obj_r=tf.reshape(obj_r,(-1,k_obj,3))
    
    return [cam_t,cam_p,cam_r],[obj_t,obj_p, obj_r,obj_mask]

def motion_net(input_frame_0,input_frame_1,point_cloud_0,reuse=False):
    with tf.variable_scope('motion_net',reuse=reuse):
        cam_motion,obj_motion=param_net(input_frame_0,input_frame_1,k_obj=4,)
        pix_pos,flow,point_cloud,motion_map=Optical_transformer()(point_cloud_0,cam_motion,obj_motion,)
        return pix_pos,flow,point_cloud,motion_map