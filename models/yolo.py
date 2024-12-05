# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import *
from models.experimental import *
from models.org_cosinenet import OriginalCosineNet
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

class StyleNet(nn.Module):
    # YOLO Style CLASSIFICATION HEAD
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=10, anchors=(), ch=(), inplace=True):  # classification layer
        super().__init__()
        self.num_classes = 10 # number of calsses.  1 as we only have pedestrian dataset
        self.nc = nc  # number of classes
        self.no = nc   # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

        self.classifier = nn.Sequential( 
            nn.Linear(self.no, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, self.num_classes ), ## each class bunch is for each anchor (currently only one class(pedestrian) * 3 Anchors)
        )

        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        cs = [] # classifier output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            
            ##
            bs, _, ny, nx = x[i].shape  # x(bs,3*20,10,10) to x(bs,3,10,10,20)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #here x = color sim
            cs_i = x[i].contiguous().view(-1, x[i].shape[-1])
            cs_i = self.classifier(cs_i) ## classifier
            cs_i = cs_i.contiguous().view(bs, self.na, ny, nx, -1) # (bs, num_anchors(=3), ny, nx, num_of_classes)
            cs.append(cs_i)
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    y = x[i]
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return [x, cs] if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)
    
class ColorNet(nn.Module):
    # YOLO COLOR CLASSIFICATION HEAD
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=20, anchors=(), ch=(), inplace=True):  # classification layer
        super().__init__()
        self.num_classes = 20 # number of calsses.  1 as we only have pedestrian dataset
        self.nc = nc  # number of classes
        self.no = nc   # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.classifier = nn.Sequential( 
            nn.Linear(self.no, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, self.num_classes ), ## each class bunch is for each anchor (currently only one class(pedestrian) * 3 Anchors)
        )

        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        cs = [] # classifier output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            
            ##
            bs, _, ny, nx = x[i].shape  # x(bs,3*20,10,10) to x(bs,3,10,10,20)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #here x = color sim
            cs_i = x[i].contiguous().view(-1, x[i].shape[-1])
            cs_i = self.classifier(cs_i) ## classifier
            cs_i = cs_i.contiguous().view(bs, self.na, ny, nx, -1) # (bs, num_anchors(=3), ny, nx, num_of_classes)
            cs.append(cs_i)
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    y = x[i]
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return [x, cs] if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)
    
class DirNet(nn.Module):
    # YOLO COLOR CLASSIFICATION HEAD
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=76, anchors=(), ch=(), inplace=True):  # classification layer
        super().__init__()
        self.num_classes = 76 # number of calsses.  1 as we only have pedestrian dataset
        self.nc = nc  # number of classes
        self.no = nc   # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.classifier = nn.Sequential( 
            nn.Linear(self.no, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, self.num_classes ), ## each class bunch is for each anchor (currently only one class(pedestrian) * 3 Anchors)
        )

        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        cs = [] # classifier output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            
            ##
            bs, _, ny, nx = x[i].shape  # x(bs,3*20,10,10) to x(bs,3,10,10,20)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #here x = color sim
            cs_i = x[i].contiguous().view(-1, x[i].shape[-1])
            cs_i = self.classifier(cs_i) ## classifier
            cs_i = cs_i.contiguous().view(bs, self.na, ny, nx, -1) # (bs, num_anchors(=3), ny, nx, num_of_classes)
            cs.append(cs_i)
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    y = x[i]
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return [x, cs] if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)
    
class REIDNET(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=2048, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.num_classes = 1 # number of calsses.  1 as we only have pedestrian dataset
        self.nc = nc  # number of features
        self.no = nc   # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        # self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        # self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        # self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m = nn.ModuleList(nn.Sequential(
        #                                 nn.Conv2d(x, self.no * self.na, 1),
        #                                 nn.Linear(512, 256),
        #                                 nn.Dropout(),
        #                                 nn.Linear(256, self.num_classes)
        #                             ) for x in ch
        #                 )
        self.classifier = nn.Sequential( 
            nn.Linear(self.no, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, self.num_classes ), ## each class bunch is for each anchor (currently only one class(pedestrian) * 3 Anchors)
        )

        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        cs = [] # classifier output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            
            ##
            bs, _, ny, nx = x[i].shape  # x(bs,3*512,20,20) to x(bs,3,20,20,512)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #here x = cosine_similarity
            cs_i = x[i].contiguous().view(-1, x[i].shape[-1])
            cs_i = self.classifier(cs_i) ## classifier
            cs_i = cs_i.contiguous().view(bs, self.na, ny, nx, -1) # (bs, num_anchors(=3), ny, nx, num_of_classes(1 only pedestrian))
            cs.append(cs_i)
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    y = x[i]
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return [x, cs] if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)
    
class CosineNet(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=512, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.num_classes = 1 # number of calsses.  1 as we only have pedestrian dataset
        self.nc = nc  # number of features
        self.no = nc   # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        # self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        # self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        # self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # self.m = nn.ModuleList(nn.Sequential(
        #                                 nn.Conv2d(x, self.no * self.na, 1),
        #                                 nn.Linear(512, 256),
        #                                 nn.Dropout(),
        #                                 nn.Linear(256, self.num_classes)
        #                             ) for x in ch
        #                 )
        self.classifier = nn.Sequential( 
            nn.Linear(self.no, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, self.num_classes ), ## each class bunch is for each anchor (currently only one class(pedestrian) * 3 Anchors)
        )

        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        cs = [] # classifier output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            
            ##
            bs, _, ny, nx = x[i].shape  # x(bs,3*512,20,20) to x(bs,3,20,20,512)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous() #here x = cosine_similarity
            cs_i = x[i].contiguous().view(-1, x[i].shape[-1])
            cs_i = self.classifier(cs_i) ## classifier
            cs_i = cs_i.contiguous().view(bs, self.na, ny, nx, -1) # (bs, num_anchors(=3), ny, nx, num_of_classes(1 only pedestrian))
            cs.append(cs_i)
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    y = x[i]
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return [x, cs] if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    
class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', org_cosinenet_path="org_cosinenet.t7", ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])
        ## original cosinenet load model
        # self.org_cosinenet = self.load_original_CosineNet(org_cosinenet_path)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.tensor([s / x.shape[-2] for (x, _, _, _, _, _, _, _, _) in zip(*self.forward(torch.zeros(1, ch, s, s)))]) # forward (if we output but cosinenet class and sim)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                # yi = self.forward_once(xi)[0]  # forward
                #forward with CosineNet
                y_all, cosine_sim, cs = self.forward_once(xi)
                yi = y_all[0]
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            # return torch.cat(y, 1), None  # augmented inference, train
            return (torch.cat(y, 1), cosine_sim, cs)  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            ## Integrate CosineNet
            if (type(m).__name__ == 'CosineNet' or type(m).__name__ == 'REIDNET') and self.training:
                cosine_sim, cs = m(x)
            if (type(m).__name__ == 'CosineNet' or type(m).__name__ == 'REIDNET') and not self.training:
                cosine_sim = m(x)
                cs = None
            ## Integrate color
            if (type(m).__name__ == 'ColorNet' ) and self.training:
                color_sim, color_class = m(x)
            if (type(m).__name__ == 'ColorNet' ) and not self.training:
                color_sim = m(x)
                color_class = None
            
            ## Integrate style
            if (type(m).__name__ == 'StyleNet' ) and self.training:
                style_sim, style_class = m(x)
            if (type(m).__name__ == 'StyleNet' ) and not self.training:
                style_sim = m(x)
                style_class = None
                
            ## Integrate Dir
            if (type(m).__name__ == 'DirNet' ) and self.training:
                dir_sim, dir_class = m(x)
            if (type(m).__name__ == 'DirNet' ) and not self.training:
                dir_sim = m(x)
                dir_class = None
                
            # Normal layers + Detect layer
            if (type(m).__name__ != 'CosineNet' and
                type(m).__name__ != 'REIDNET'   and
                type(m).__name__ != 'ColorNet' and
                type(m).__name__ != 'StyleNet' and
                type(m).__name__ != 'DirNet'):
                x = m(x)  
            
            # x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        # return x
        return (x, cosine_sim, cs, color_sim, color_class, style_sim, style_class, dir_sim, dir_class)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)
        
    def load_original_CosineNet(self, org_cosinenet_path):
        print( "---------------------------- Loadign Original CosineNet for GT Generation ---------------------------------")
        net = OriginalCosineNet(reid=True)
        state_dict = torch.load(org_cosinenet_path, map_location=lambda storage, loc: storage)['net_dict']
        net.load_state_dict(state_dict)
        net.to('cuda').eval()
        print( "---------------------------- Original CosineNet Warmup ---------------------------------")
        x = torch.ones(1,3,128,64).cuda()
        y = net(x)
        assert y.shape[-1] ==512, 'original CosineNet output has to have 512 dimensions' # check
        return net


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw, n_features, n_colors, n_styles, n_dirs = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['n_features'], d['n_colors'], d['n_styles'], d['n_dirs']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CompressorBlock2, CompressorBlock1, CrossConv, BottleneckCSP,
                 CompressorBlock3, C3, Down]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in {Detect, CosineNet, REIDNET, ColorNet, StyleNet, DirNet}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/phrd_cosinenet.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', default=1, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--org_cosinenet_path', default='org_cosinenet.t7', help='original cosinenet path')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg, opt.org_cosinenet_path).to(device)
    # model.train()

    # Inference
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model.eval()
    x, cosine_sim, _ =  model(im) #x -> tuple (converted x list to toch, x list), cosine_sim->tuple (converted cosine_sim list to toch, cosine_sim list)
    
    # Train
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model.train()
    x, cosine_sim, cs =  model(im) #x -> list (bs, n_anchors, nx, ny, class + 5), cosine_sim->list (cbs, n_anchors, nx, ny, 512), cs -> list (bs, n_anchors, nx, ny, 1)
    
    
    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
