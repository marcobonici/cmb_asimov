from __future__ import annotations

import unittest

import numpy as np

from cmb_asimov import CMBAsimovGenerator, Calibration


class CMBAsimovTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = CMBAsimovGenerator()
        cls.ell = np.arange(9001, dtype=np.float64)
        cls.dl_tt = 3000.0 * np.exp(-((cls.ell - 200.0) ** 2) / 20000.0)
        cls.dl_ee = 0.05 * np.exp(-((cls.ell - 200.0) ** 2) / 20000.0)
        cls.dl_te = 100.0 * np.exp(-((cls.ell - 300.0) ** 2) / 25000.0)
        cls.dl_tt[:2] = cls.dl_ee[:2] = cls.dl_te[:2] = 0.0

    def test_primary_shapes_and_finiteness(self):
        result = self.generator.primary_cmb(self.dl_tt, self.dl_te, self.dl_ee)
        self.assertEqual(result["camspec"].shape, (3013,))
        self.assertEqual(result["act"].shape, (109,))
        self.assertEqual(result["spt"].shape, (196,))
        for vector in result.values():
            self.assertTrue(np.all(np.isfinite(vector)))

    def test_camspec_selection_and_calibration(self):
        calibration = Calibration(A_planck=1.01, calTE=0.99, calEE=1.02)
        result = self.generator.primary_cmb(self.dl_tt, self.dl_te, self.dl_ee, calibration)[
            "camspec"
        ]
        a2 = calibration.A_planck**2
        expected = np.concatenate(
            (
                self.dl_tt[30:1501] / a2,
                self.dl_te[30:1001] / (a2 * calibration.calTE),
                self.dl_ee[30:601] / (a2 * calibration.calEE),
            )
        )
        np.testing.assert_array_equal(result, expected)

    def test_act_contains_inclusive_boundary_bins(self):
        self.assertAlmostEqual(self.generator.primary["act_effective_ell_tt"][0], 1500.5)
        self.assertAlmostEqual(self.generator.primary["act_effective_ell_te"][0], 1000.5)
        self.assertAlmostEqual(self.generator.primary["act_effective_ell_ee"][0], 600.5)

    def test_data_coordinates_match_vector_blocks(self):
        coordinates = self.generator.coordinates()
        self.assertEqual(coordinates["camspec_ell_tt"].shape, (1471,))
        self.assertEqual(coordinates["camspec_ell_te"].shape, (971,))
        self.assertEqual(coordinates["camspec_ell_ee"].shape, (571,))
        self.assertEqual(coordinates["act_effective_ell_tt"].shape, (27,))
        self.assertEqual(coordinates["act_effective_ell_te"].shape, (37,))
        self.assertEqual(coordinates["act_effective_ell_ee"].shape, (45,))
        self.assertEqual(coordinates["spt_effective_ell"].shape, (196,))
        self.assertEqual(coordinates["lensing_bcents_act"].shape, (10,))
        self.assertEqual(coordinates["lensing_bcents_planck"].shape, (9,))
        self.assertEqual(coordinates["lensing_bcents_spt"].shape, (16,))

    def test_compact_lensing_regression(self):
        ell = np.arange(4001, dtype=np.float64)
        cl_pp = np.zeros(4001)
        cl_pp[2:] = 1e-2 / ((ell[2:] + 10.0) * (ell[2:] + 11.0)) ** 3

        cl_tt = np.zeros(4001)
        cl_te = np.zeros(4001)
        cl_ee = np.zeros(4001)
        cl_bb = np.zeros(4001)
        factor = 2.0 * np.pi / (ell[2:] * (ell[2:] + 1.0))
        cl_tt[2:] = factor * 2000.0 * np.exp(-((ell[2:] - 200.0) ** 2) / 20000.0)
        cl_te[2:] = factor * 100.0 * np.exp(-((ell[2:] - 300.0) ** 2) / 25000.0)
        cl_ee[2:] = factor * 0.05 * np.exp(-((ell[2:] - 200.0) ** 2) / 20000.0)

        expected = np.array(
            [
                -1.1247922629044855e-07,
                -1.8332200455703184e-07,
                -1.8625752006922753e-07,
                -1.6082220895940642e-07,
                -1.3137695549846316e-07,
                -1.0588104476789644e-07,
                -8.341864085628456e-08,
                -6.5440216822958e-08,
                -5.2158329960301626e-08,
                -4.196029859131253e-08,
                5.22214114968428e-08,
                -1.3712644405753935e-07,
                -1.8338602141936357e-07,
                -1.6847448441401413e-07,
                -1.425940123606623e-07,
                -1.2442879690434596e-07,
                -1.0455245751561413e-07,
                -9.389450289931313e-08,
                -7.914271761032749e-08,
                4.76921902096727e-07,
                3.0413762963212073e-07,
                1.862599384438044e-07,
                1.1388783907538447e-07,
                7.48611078103343e-08,
                4.897277912637599e-08,
                3.120365741824643e-08,
                1.9345013111394282e-08,
                1.1874589753392185e-08,
                7.364312170049e-09,
                4.5109083820802855e-09,
                2.7393930396310538e-09,
                1.6530280473252888e-09,
                9.960007398206805e-10,
                6.006088179970855e-10,
                3.5679239850360174e-10,
            ]
        )
        result = self.generator.lensing_cmb(cl_pp, cl_tt, cl_te, cl_ee, cl_bb)
        np.testing.assert_allclose(result, expected, rtol=2e-13, atol=1e-18)

    def test_short_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 8502"):
            self.generator.primary_cmb(np.zeros(10), np.zeros(10), np.zeros(10))
        with self.assertRaisesRegex(ValueError, "at least 3102"):
            self.generator.lensing_cmb(
                np.zeros(10),
                np.zeros(3000),
                np.zeros(3000),
                np.zeros(3000),
                np.zeros(3000),
            )


if __name__ == "__main__":
    unittest.main()
