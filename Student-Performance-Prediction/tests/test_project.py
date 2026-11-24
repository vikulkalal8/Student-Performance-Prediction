import io, json, unittest, zipfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from model import available, experiment, predict, UCI
from fetch_dataset import extract
class ProjectTests(unittest.TestCase):
    def test_original_results(self):
        r,_=experiment('synthetic'); self.assertEqual((r['rows'],r['train'],r['test']),(20,16,4)); self.assertEqual(r['selected'],'2'); self.assertAlmostEqual(r['comparison'][2]['mse'],0.76367226,places=6)
    def test_pca_full_equivalent(self):
        for mode in available():
            r,_=experiment(mode); np.testing.assert_allclose(r['predictions']['baseline'],r['predictions'][str(len(r['features']))],rtol=1e-7,atol=1e-7); self.assertAlmostEqual(sum(r['explained']),100)
            np.testing.assert_allclose(np.array(r['covariance'])@np.array(r['eigenvectors']),np.array(r['eigenvectors'])*np.array(r['eigenvalues']),atol=1e-8)
    def test_prediction_and_validation(self):
        r,_=experiment(); values={f:b['default'] for f,b in r['bounds'].items()}; pred=predict('synthetic','2',values); self.assertAlmostEqual(pred['prediction'],pred['intercept']+np.dot(pred['pcs'],pred['coefficients'])); values['Attendance']=101
        with self.assertRaises(ValueError): predict('synthetic','2',values)
    def test_nested_zip_import(self):
        inner=io.BytesIO(); outer=io.BytesIO()
        with zipfile.ZipFile(inner,'w') as z:z.writestr('student-mat.csv',b'test')
        with zipfile.ZipFile(outer,'w') as z:z.writestr('student.zip',inner.getvalue())
        self.assertEqual(extract(outer.getvalue()),b'test')
    def test_uci_pipeline_schema(self):
        # Fabricated fixture ONLY for testing the UCI code path; never saved as real data.
        rng=np.random.default_rng(42); frame=pd.DataFrame({f:rng.integers(1,5,395) for f in UCI}); frame['G3']=rng.integers(0,21,395)
        import model
        original=model.load
        def fixture(mode):
            if mode.startswith('uci'):
                features=model.SPECS[mode][1]; return frame[features+['G3']],features,'G3',20
            return original(mode)
        with patch('model.load',side_effect=fixture):
            for mode in ['uci','uci_early']:
                experiment.cache_clear(); r,_=experiment(mode); self.assertEqual(r['rows'],395); self.assertEqual(r['test'],79)
                values={f:round(b['default']) for f,b in r['bounds'].items()}; self.assertTrue(np.isfinite(predict(mode,r['selected'],values)['prediction']))
        experiment.cache_clear()
if __name__=='__main__': unittest.main()
