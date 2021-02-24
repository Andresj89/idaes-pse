import pyomo.environ as pyo
import idaes.core.util.convergence.convergence_base as cb
import supercritical_powerplant_simple_helm_2 as scpp

class scppConvergenceEvaluation(cb.ConvergenceEvaluation):

    def get_specification(self):

        s = cb.ConvergenceEvaluationSpecification()

        s.add_sampled_input(
            name='Inlet flow',
            pyomo_path='m.fs.boiler.inlet.flow_mol',
            lower=20000, upper=40000, std=10000
        )
        return s

    def get_initialized_model(self):

        m = scpp.build_plant_model(initialize_from_file=None,
                          store_initialization=None)

        opt = self.get_solver()
        opt.solve(m, tee=True, symbolic_solver_labels=True)
        print('Total Power =', pyo.value(m.fs.plant_power_out[0]))
        return m

    def get_solver(self):

        solver = pyo.SolverFactory("ipopt")
        solver.options = {
            "tol": 1e-6,
            "max_iter": 300,
            "halt_on_ampl_error": "yes",
        }
        return solver

# test = scppConvergenceEvaluation()
# test.get_initialized_model()
# scppConvergenceEvaluation.get_initialized_model(cb.ConvergenceEvaluation)