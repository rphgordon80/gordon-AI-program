# gordon-AI-program
AI course 
This is the full error and stack dump that occured.  My two networks, vgg and resnet, seem to train correctly but when I did the Sanity check using the
model.forward passing in a single image, it generated the error below.


-------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-64-f9060237a5aa> in <module>()
      8 
      9 
---> 10 top_p,top_class  = predict(image_path,model)
     11 #top_p
     12 #top_class

<ipython-input-61-1b7a575a5759> in predict(image_path, model)
     21     with torch.no_grad():
     22         #image=process_image(image)
---> 23         output = model.forward(image)
     24         output.to(device)
     25         print('inside predict after forward',device)

/opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/models/vgg.py in forward(self, x)
     42         x = self.features(x)
     43         x = x.view(x.size(0), -1)
---> 44         x = self.classifier(x)
     45         return x
     46 

/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
    489             result = self._slow_forward(*input, **kwargs)
    490         else:
--> 491             result = self.forward(*input, **kwargs)
    492         for hook in self._forward_hooks.values():
    493             hook_result = hook(self, input, result)

/opt/conda/lib/python3.6/site-packages/torch/nn/modules/container.py in forward(self, input)
     89     def forward(self, input):
     90         for module in self._modules.values():
---> 91             input = module(input)
     92         return input
     93 

/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py in __call__(self, *input, **kwargs)
    489             result = self._slow_forward(*input, **kwargs)
    490         else:
--> 491             result = self.forward(*input, **kwargs)
    492         for hook in self._forward_hooks.values():
    493             hook_result = hook(self, input, result)

/opt/conda/lib/python3.6/site-packages/torch/nn/modules/linear.py in forward(self, input)
     53 
     54     def forward(self, input):
---> 55         return F.linear(input, self.weight, self.bias)
     56 
     57     def extra_repr(self):

/opt/conda/lib/python3.6/site-packages/torch/nn/functional.py in linear(input, weight, bias)
    990     if input.dim() == 2 and bias is not None:
    991         # fused op is marginally faster
--> 992         return torch.addmm(bias, input, weight.t())
    993 
    994     output = input.matmul(weight.t())

RuntimeError: size mismatch, m1: [1 x 14336], m2: [25088 x 4096] at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCTensorMathBlas.cu:249
