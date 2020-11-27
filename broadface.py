import torch
def get_class_center(class_center,target):
    '''
    通过标签target 从 类别代理矩阵class_center 提取对应类中心
    :param class_center:类别代表矩阵 [类别总数,feature_dim]
    :param target: 标签 [batch_size]
    :return: [batch_size,feature_dim]
    '''
    return class_center[target.long()]

def feature_compensate(feature, class_center, update_class_center):
    '''
    补偿特征   论文式（7）
    作用：通过 当前类中心与过去类中心的差值，将 过去特征 补偿近似为 当前特征
    :param feature: 队列内过去的特征  [batch * queue_nums,feature_dim]
    :param class_center: 队列内过去的类中心  [batch * queue_nums,feature_dim]
    :param update_class_center: 队列内对应的当前类中心 [batch * queue_nums,feature_dim]
    :return: [batch * queue_nums,512]
    '''
    class_center = class_center.to(feature.device)
    update_class_center = update_class_center.to(feature.device)
    # [batch * queue_nums,1]
    lamuda = (torch.norm(feature, dim=1) / torch.norm(class_center, dim=1)).unsqueeze(-1)
    return feature + lamuda * (update_class_center - class_center)

def start_train():
    '''
    BroadFace
    '''
    ######################
    batch_size=120
    queue_nums=9 # 队列大小
    ######################
    init_param=None
    optimizer = init_param['optimizer'] # 优化器
    lr_scheduler = init_param['lr_scheduler'] # 学习率调度器
    face_model = init_param['face_model'] # Backbone
    classifier = init_param['classifier'] # 分类器
    face_data = init_param['face_data'] # 数据加载器


    # 队列存储 特征\对应类中心\对应标签
    feature_List, class_center_List, label_List = [],[],[]
    # 开始训练
    for epoch in range(0, 100):
        optimizer.zero_grad()
        for batch_idx, (input_imgs, target_class) in enumerate(face_data):

            input_imgs = input_imgs.cuda()
            target_class = target_class.cuda()

            # 获取指定类别的当前类中心[batch,feature_dim]
            # classifier.model.id_agent.data  :分类器内部的类别代理矩阵id_agent
            batch_class_center = get_class_center(classifier.model.id_agent.data,target_class)
            if len(feature_List)<queue_nums:
                batch_feature = face_model(input_imgs)
                optimizer.zero_grad() # 无需更新参数，故梯度清零
                feature_List.append(batch_feature.detach()) # 断开梯度
                class_center_List.append(batch_class_center.detach())
                label_List.append(target_class.detach())
            else:
                feature= torch.cat(feature_List, dim=0)  # [batch * queue_nums,512]
                class_center = torch.cat(class_center_List, dim=0)  # [batch * queue_nums,512]
                label=torch.cat(label_List, dim=0) # [batch * queue_nums]

                # 获取指定类别的当前类中心[batch * queue_nums,512]
                # classifier.model.id_agent.data  :分类器内部的类别代理矩阵id_agent
                update_class_center = get_class_center(classifier.model.id_agent.data,label)

                # 补偿特征，更新类中心[batch * queue_nums,512]
                compensated_feature = feature_compensate(feature, class_center, update_class_center)

                batch_feature = face_model(input_imgs)
                # cat操作前后区分梯度. compensated_feature无梯度，batch_feature有梯度  -> feature_input有梯度
                feature_input = torch.cat((compensated_feature, batch_feature), 0)
                # 无梯度
                target_input = torch.cat((label, target_class), 0)
                outputs = classifier((feature_input, target_input))
                loss_750k = outputs['losses'].mean()
                loss_750k.backward()
                optimizer.step()
                optimizer.zero_grad()


                # ==========更新队列================
                # 替换为 补偿后的特征
                feature_List= list(compensated_feature.chunk(queue_nums,0))
                # 替换为  当前类中心
                class_center_List= list(update_class_center.chunk(queue_nums,0))

                # 队列去掉最远batch
                feature_List.pop(0)
                class_center_List.pop(0)
                label_List.pop(0)

                # 队列加入当前batch
                feature_List.append(batch_feature.detach())
                class_center_List.append(batch_class_center.detach())
                label_List.append(target_class.detach())

                # 评估、保存模型
        lr_scheduler.step()