"""Apply the supplied causal-window contract to the optional local model."""
import torch
import test_contract as contract
from student import StudentGPT
from student_local import LocalGPT


class LocalContractTests(contract.ContractTests):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.config=dict(vocab=2048,width=32,heads=4,depth=2,context=256,local_kernel=3)
        self.model=LocalGPT(self.config).eval()
        # Exercise actual learned neighboring-token mixing, not only identity.
        with torch.no_grad():
            for block in self.model.blocks:block.local_qkv.weight.normal_(0,.3)

    def test_identity_convolution_preserves_original_predictions(self):
        reference=StudentGPT(self.config).eval()
        local=LocalGPT(self.config).eval()
        result=local.load_state_dict(reference.state_dict(),strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertEqual(len(result.missing_keys),self.config['depth'])
        x=torch.randint(2048,(2,32))
        with torch.no_grad():
            torch.testing.assert_close(local.predict_log_probs(x),reference.predict_log_probs(x),
                                       atol=1e-6,rtol=1e-6)

    def test_fused_inference_matches_differentiable_convolution(self):
        x=torch.randint(2048,(2,32))
        reference=self.model.predict_log_probs(x).detach()
        with torch.no_grad():
            actual=self.model.predict_log_probs(x)
        torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-6)
