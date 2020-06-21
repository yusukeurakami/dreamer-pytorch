    
import torch


sample = self._dist.sample_n(self._samples)
print("sample: ", sample.size()) # torch.Size([100, 1, 6])
logprob = self._dist.log_prob(sample)
print("logprob: ",logprob.size())
# still not sure how the following parts work
logprob_argmax = torch.argmax(logprob,0)
print("logprob argmax:", logprob_argmax.size()) # torch.Size([1, 6])
sample = sample[logprob_argmax]
print("sample selected: ",sample.size()) #torch.Size([1, 6, 1, 6]) --> have to be [1, 6]