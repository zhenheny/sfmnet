class Cloud_transformer():
    def __init__(self,intrinsics=[0.5,0.5,1.0], **kwargs):
        self.output_dim = 3
        self.cam_intrinsics = intrinsics
        self.build()

    def build(self):
        self.cx_=self.cam_intrinsics[0]
        
        self.cy_=self.cam_intrinsics[1]
        
        self.cf_=self.cam_intrinsics[2]
        
        self.cx=tf.constant(self.cam_intrinsics[0],dtype=tf.float32)
        
        self.cy=tf.constant(self.cam_intrinsics[1],dtype=tf.float32)
        
        self.cf=tf.constant(self.cam_intrinsics[2],dtype=tf.float32)
        
    def mesh_grid(self,width,height):
        # get 
        """
        [(xi/w-cx)/f,(yi/h-cy)/f,1]
        
        next just 
        
        d*[(xi/w-cx)/f,(yi/h-cy)/f,1]
         to get [Xi,Yi,Zi]
        
        """
      
        x_linspace=tf.linspace(-self.cx_,1-self.cx_,width)
        y_linspace=tf.linspace(-self.cy_,1-self.cy_,height)
        
#         x_cord,y_cord=tf.meshgrid(x_linspace,y_linspace)
        y_cord,x_cord=tf.meshgrid(y_linspace,x_linspace)
        
        
        x_cord=tf.reshape(x_cord,[-1])
        y_cord=tf.reshape(y_cord,[-1])
        
        f_=tf.ones_like(x_cord)
        
        x_=tf.div(x_cord,self.cf)
        y_=tf.div(y_cord,self.cf)
        
        grid=tf.concat([x_,y_,f_],0)
        return grid
        
    def transform(self,x):
        
        #get input shape
        batch_size=tf.shape(x)[0]
        width=tf.shape(x)[1]
        height=tf.shape(x)[2]
        channel=tf.shape(x)[3]
        batch_size=tf.cast(batch_size,tf.int32)
        width=tf.cast(width,tf.int32)
        height=tf.cast(height,tf.int32)
        channel=tf.cast(channel,tf.int32)
        
        
        #grid
        grid=self.mesh_grid(width,height)
        grid=tf.expand_dims(grid,0)
        grid=tf.reshape(grid,[-1])
        
        
        grid_stack = tf.tile(grid, tf.stack([batch_size]))
        grid_stack=tf.reshape(grid_stack,[batch_size,3,-1])
        depth=tf.reshape(x,[batch_size,1,-1])
        depth=tf.concat([depth]*self.output_dim,1)
        
        point_cloud=tf.multiply(depth,grid_stack)
#         pc3=tf.reshape(pc3,[batch_size,width,height,self.output_dim])
        
        return point_cloud

    def __call__(self, x):
        point_cloud=self.transform(x)
        return point_cloud


class Optical_transformer():

    def __init__(self,intrinsics=[0.5,0.5,1.0],img_shape=[384,128], **kwargs):
        
        self.cam_intrinsics = intrinsics
        self.img_w=self.np_tf(img_shape[0])
        self.img_h=self.np_tf(img_shape[1])
        self.img_w_=float(img_shape[0])
        self.img_h_=float(img_shape[1])
        
        
        
        self.cx_=self.cam_intrinsics[0]
        
        self.cy_=self.cam_intrinsics[1]
        
        self.cf_=self.cam_intrinsics[2]
        
        
        
        self.cx=self.np_tf(self.cam_intrinsics[0])
        
        self.cy=self.np_tf(self.cam_intrinsics[1])
        
        self.cf=self.np_tf(self.cam_intrinsics[2])
        

        so3_a=np.array([
            [0,-1,0,1,0,0,0,0,0],
            [1,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,1]
        ])

        so3_b=np.array([
            [0,0,1,0,0,0,-1,0,0],
            [1,0,0,0,0,0,0,0,1],
            [0,0,0,0,1,0,0,0,0]
        ])

        so3_y=np.array([
            [0,0,0,0,0,-1,0,1,0],
            [0,0,0,0,1,0,0,0,1],
            [1,0,0,0,0,0,0,0,0]
        ])

#         so3_param=np.concatenate([so3_a,so3_b,so3_y],0)
        self.so3_a=self.np_tf(so3_a)
        self.so3_b=self.np_tf(so3_b)
        self.so3_y=self.np_tf(so3_y)
        
  
    def np_tf(self,array):
        return tf.constant(array,tf.float32)
    

    def build(self,cam_motion,obj_motion,x):
                
        self.cam_motion=cam_motion
        
        self.obj_motion=obj_motion
        self.mask_size=obj_motion[0].shape.as_list()[1]
        self.x_shape=x.shape.as_list()
        
    
#    tranformation     
    def so3_mat(self,sin):
        #input :sin a,sin b,sin y 
        #return : SO3
        sin=tf.expand_dims(sin,-1)
        cos=tf.sqrt(tf.ones_like(sin)-tf.square(sin))
        t=tf.concat([sin,cos,tf.ones_like(sin)],-1)
        t_a=tf.slice(t,[0,0,0],[-1,1,-1])
        t_b=tf.slice(t,[0,1,0],[-1,1,-1])
        t_y=tf.slice(t,[0,2,0],[-1,1,-1])
        t_a=tf.reshape(t_a,(-1,3))
        t_b=tf.reshape(t_b,(-1,3))
        t_y=tf.reshape(t_y,(-1,3))
        
        soa=tf.matmul(t_a,self.so3_a)
        soa=tf.reshape(soa,(-1,3,3))
        
        sob=tf.matmul(t_b,self.so3_b)
        sob=tf.reshape(sob,(-1,3,3))
        soy=tf.matmul(t_y,self.so3_y)
        soy=tf.reshape(soy,(-1,3,3))
   
        so3=tf.matmul(soa,  tf.matmul(sob,soy))
        return so3
    def pior_pont(self,p):
        batch_size=p.shape.as_list()[0]
        p_ret=tf.reshape(p,(-1,30,20))
        p_y=tf.reduce_sum(p_ret,1)
        p_x=tf.reduce_sum(p_ret,2)
        x_loc=tf.linspace(-30.0,30.0,30)
        y_loc=tf.linspace(-20.,20.,20)
        P_x_loc=tf.reduce_mean(tf.multiply(p_x,x_loc))
        P_x_loc=tf.reshape(P_x_loc,(-1,1))
        P_y_loc=tf.reduce_mean(tf.multiply(p_y,y_loc))
        P_y_loc=tf.reshape(P_y_loc,(-1,1))
        
        ground=tf.ones_like(P_y_loc)
        P=tf.concat([P_x_loc,P_y_loc,ground],1)
        
        
        return P
    
    def rigid_motion(self,x,R,p,t):
        p=tf.expand_dims(p,-1)
        
        t=tf.expand_dims(t,-1)
        
        motion=tf.add(tf.matmul(R,tf.subtract(x,p)),t)

        return motion
          
    
    def cam_motion_transform(self,x):
        t,p,sin=self.cam_motion
        p=self.pior_pont(p)
        R=self.so3_mat(sin)
        X=self.rigid_motion(x,R,p,t)
        
        
        return X
        
  
    def obj_motion_transform(self,x_input):
        t,p,sin,mask=self.obj_motion
        p=self.pior_pont(p)
        sin=tf.reshape(sin,(-1,3))
        p=tf.reshape(p,(-1,3))
        t=tf.reshape(t,(-1,3))
        x_in=tf.expand_dims(x_input,1)
        x_exp=tf.concat([x_in]*self.mask_size,1)
        x_=tf.reshape(x_exp,(-1,3,384*128))
        
        
        R=self.so3_mat(sin)
        
        x=self.rigid_motion(x_,R,p,t)
        
        
        
        x=tf.reshape(x,(-1,self.mask_size,3,384*128))
        x,motion_map=self.mask_motion(x,mask,x_exp)
        X=tf.add(x_input,x)
        
        
        return X,motion_map
    
    def mask_motion(self,x,mask,x_exp):
        mask=tf.reshape(mask,(-1,self.mask_size,1,384*128))
        x=tf.subtract(x,x_exp)
        motion_map=tf.multiply(x,mask)
#         x=tf.reshape(x,(-1,self.mask_size,3,384*128))
        x=tf.reduce_sum(motion_map,1)
#         print(x.shape.as_list())
        
        return x,motion_map
        
        
    def tranform_2d(self,x):
        x_3d=tf.slice(x,(0,0,0),(-1,1,49152))
        y_3d=tf.slice(x,(0,1,0),(-1,1,49152))
        z_3d=tf.slice(x,(0,2,0),(-1,1,49152))
        x_z=tf.div(x_3d,z_3d)
        y_z=tf.div(y_3d,z_3d)
        
#         x_2d=tf.multiply(self.img_w,tf.add(tf.multiply(self.cf,x_z),self.cx))
#         y_2d=tf.multiply(self.img_h,tf.add(tf.multiply(self.cf,y_z),self.cy))
        x_2d=tf.add(tf.multiply(self.cf,x_z),self.cx)
        y_2d=tf.add(tf.multiply(self.cf,y_z),self.cy)
        pos_2d_new=tf.concat([x_2d,y_2d],1)
        return pos_2d_new

    def get_flow(self,pos_2d_new):
        
        
        x_linspace = tf.linspace(0.,1.,int(self.img_w_))
        y_linspace = tf.linspace(0.,1.,int(self.img_h_))
        y_linspace,x_linspace = tf.meshgrid( y_linspace,x_linspace)
        x_linspace = tf.reshape(x_linspace, [1,-1])
        y_linspace = tf.reshape(y_linspace, [1,-1])
        pos_ori=tf.concat([x_linspace,y_linspace],0)
        flow=tf.subtract(pos_2d_new,pos_ori)
        
        
        return flow

  
    def __call__(self,x,cam_motion,obj_motion,):
        self.build( cam_motion,obj_motion,x)

        
        point_cloud,motion_map=self.obj_motion_transform(x)
        
        
        point_cloud=self.cam_motion_transform(point_cloud)
        pix_pos=self.tranform_2d(point_cloud)
        flow=self.get_flow(pix_pos)
        motion_map=tf.reshape(motion_map,(-1,img_h,img_w,1))
        return pix_pos,flow,point_cloud,motion_map

class get_frame_loss():
    def __init__(self):
        self.output_size=[128,384]
    def __call__(self,frame0,frame1,pos_2d_new,reuse=False):
        with tf.variable_scope('frame_loss',reuse=reuse):
            batch_size = tf.shape(frame1)[0]
            height = 128
            width = 384
            num_channels = 3
            output_height=128
            output_width=384



            x_s = tf.slice(pos_2d_new, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(pos_2d_new, [0, 1, 0], [-1, 1, -1])
            x_s_flatten = tf.reshape(x_s, [-1])
            y_s_flatten = tf.reshape(y_s, [-1])

            transformed_image = self._interpolate(frame1,
                                                    x_s_flatten,
                                                    y_s_flatten,
                                                    self.output_size)

            transformed_image = tf.reshape(transformed_image, shape=(-1,
                                                                    output_height,
                                                                    output_width,
                                                                     num_channels))

            loss=self.compute_loss( frame0,transformed_image )
            ####test--------------------------------------------
            return loss
    
    
    def compute_loss(self,frame0,transformed_image):
        loss=tf.reduce_mean(tf.abs(tf.subtract( frame0,transformed_image)))
        return loss
        
        
        
        
    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = 128
        width = 384
        num_channels = tf.shape(image)[3]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x =x*(width_float)
        y =y*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)
        




        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output
        
    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

class get_smooth_loss():
    def __init__(self,kernel=[[1,2,1],[0,0,0],[-1,-2,-1]],order=1):
        self.kernel=np.array(kernel)
        self.order=order
        
        
    def build(self,field_c):
        v_kernel=self.kernel
        h_kernel=self.kernel.T
        h_init=keras.initializers.Constant(value=h_kernel)
        v_init=keras.initializers.Constant(value=v_kernel)
        
        self.conv_h=Conv2D(filters=field_c,kernel_size=3,strides=1,kernel_initializer=h_init,padding='same')
        self.conv_h.trainable=False
        self.conv_v=Conv2D(filters=field_c,kernel_size=3,strides=1,kernel_initializer=v_init,padding='same')
        self.conv_v.trainable=False
        
        
    def compute_gradient(self,field):
        loss_v=self.conv_v(field)
        loss_h=self.conv_h(field)
        
        gradient_loss=loss_h+loss_v
        
        return gradient_loss
        
        
    def compute_loss(self,field):
        f1_gradient_loss=self.compute_gradient(field)
        

        
        if self.order==1:
            loss=tf.reduce_mean(tf.abs(f1_gradient_loss),-1)
            loss=tf.reduce_mean(loss)
            
        if self.order==2:
            f2_gradient_loss=self.compute_gradient(f1_gradient_loss)
            
            loss=tf.reduce_mean(tf.abs(f2_gradient_loss),-1)
            loss=tf.reduce_mean(loss)

            
        return loss
    
    def __call__(self,field,loss_type=None,reuse=False):
        with tf.variable_scope(loss_type,reuse=reuse):
            

            if loss_type=='flow':
                field=Permute((2,1))(field)
                field=tf.reshape(field,(-1,128,384,2))
                field_c=field.shape.as_list()[1]
    # #         if loss_type=='depth':
    # #             field=field#tf.reshape(field,(-1,128,384))
    # #             field_c=1
    #         else:

            field_c=field.shape.as_list()[-1]

            self.build(field_c)
            loss=self.compute_loss(field)
            return loss


class get_fb_depth_loss():
    def __init__(self):
        self.output_size=[128,384]
    def __call__(self,depth0,depth1,pos_2d_new,motion,reuse=False):
        with tf.variable_scope('fb_depth_loss',reuse=reuse):
            batch_size = tf.shape(depth0)[0]
            height = tf.shape(depth0)[1]
            width = tf.shape(depth0)[2]
            num_channels = tf.shape(depth0)[3]
            output_height=self.output_size[0]
            output_width=self.output_size[1]



            x_s = tf.slice(pos_2d_new, [0, 0, 0], [-1, 1, -1],name='err')
            y_s = tf.slice(pos_2d_new, [0, 1, 0], [-1, 1, -1])
            x_s_flatten = tf.reshape(x_s, [-1])
            y_s_flatten = tf.reshape(y_s, [-1])

            transformed_depth1 = self._interpolate(depth1,
                                                    x_s_flatten,
                                                    y_s_flatten,
                                                    self.output_size)

            transformed_depth1 = tf.reshape(transformed_depth1, shape=(-1,
                                                                    output_height,
                                                                    output_width,
                                                                     num_channels))

            motion_z=tf.slice(motion,[0,2,0],[-1,1,-1])
            motion_z=tf.reshape(motion_z,(-1,output_height,output_width,1))
            transformed_depth0=tf.add(depth0,motion_z)
            loss=self.compute_loss( transformed_depth0,transformed_depth1 )
        return loss
    
    
    def compute_loss(self,transformed_depth0,transformed_depth1):
        
        loss=tf.reduce_mean(tf.abs(tf.subtract( transformed_depth0,transformed_depth1)))
        return loss
        
        
        
        
    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x =x*(width_float)
        y = y*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output
        
    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

def unpack_image_sequence(self, image_seq):
    img_width, img_height = 384, 128
    num_source = 3
    # Assuming the center image is the target frame
    tgt_start_idx = int(img_width * (num_source//2))
    tgt_image = tf.slice(image_seq, 
                         [0, tgt_start_idx, 0], 
                         [-1, img_width, -1])
    # Source fames before the target frame
    src_image_1 = tf.slice(image_seq, 
                           [0, 0, 0], 
                           [-1, int(img_width * (num_source//2)), -1])
    # Source frames after the target frame
    src_image_2 = tf.slice(image_seq, 
                           [0, int(tgt_start_idx + img_width), 0], 
                           [-1, int(img_width * (num_source//2)), -1])
    src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
    # Stack source frames along the color channels (i.e. [H, W, N*3])
    src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                [0, i*img_width, 0], 
                                [-1, img_width, -1]) 
                                for i in range(num_source)], axis=2)
    src_image_stack.set_shape([img_height, 
                               img_width, 
                               num_source * 3])
    tgt_image.set_shape([img_height, img_width, 3])
    return tgt_image, src_image_stack

def format_file_list(self, data_root, split):
    with open(data_root + '/%s.txt' % split, 'r') as f:
        frames = f.readlines()
    subfolders = [x.split(' ')[0] for x in frames]
    frame_ids = [x.split(' ')[1][:-1] for x in frames]
    image_file_list = [os.path.join(data_root, 'frame_seq_384_128', subfolders[i], 
        frame_ids[i] + '.jpg') for i in range(len(frames))]
    date = [x.split('_drive')[0] for x in frames]
    folders = [x.split('_sync')[0]+"_sync" for x in frames]
    cam_file_list = [os.path.join(data_root, 'frame_seq_384_128', subfolders[i], 
        frame_ids[i] + '_cam.txt') for i in range(len(frames))]
    all_list = {}
    all_list['image_file_list'] = image_file_list
    all_list['cam_file_list'] = cam_file_list
    return all_list