import torch
import test_contract as contract
from student_hash import HashGPT
from student_local import LocalGPT


class HashContractTests(contract.ContractTests):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.config=dict(vocab=2048,width=32,heads=4,depth=2,context=256,
                         local_kernel=3,hash_buckets=128,hash_width=8)
        self.model=HashGPT(self.config).eval()

    def test_short_prefixes_and_hash_gradient(self):
        for length in [1,2,3]:
            x=torch.randint(2048,(2,length))
            self.assertEqual(tuple(self.model(x).shape),(2,length,2048))
        x=torch.randint(2048,(2,12));self.model(x).square().mean().backward()
        for table in self.model.hash_tables:
            self.assertGreater(table.weight.grad.abs().sum().item(),0)

    def test_zero_hash_projection_recovers_core(self):
        core=LocalGPT(self.config).eval()
        state={k:v for k,v in self.model.state_dict().items() if not k.startswith('hash_')}
        core.load_state_dict(state)
        with torch.no_grad():
            self.model.hash_projection.weight.zero_()
            x=torch.randint(2048,(2,24))
            torch.testing.assert_close(self.model(x),core(x),atol=1e-6,rtol=1e-6)
