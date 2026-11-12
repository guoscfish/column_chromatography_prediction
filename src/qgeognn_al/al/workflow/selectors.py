"""Profile selection adapters; the RoundWorkflow alone owns freeze and labels."""
from ...infra.contracts import content_hash
from ..numerical_profiles import numerical_profile, select_numerical_profile


class NumericalSelector:
    def __init__(self, profile, *, original_candidate_ids, reference_ids, backend,
                 validation_ids, validation_labels):
        self.profile = numerical_profile(profile)
        self.original_candidate_ids = tuple(original_candidate_ids)
        self.reference_ids = tuple(reference_ids)
        self.backend = backend
        self.validation_ids,self.validation_labels = tuple(validation_ids),dict(validation_labels)

    @property
    def contract_hash(self):
        return content_hash({'profile':self.profile.to_dict(),'original_U0':self.original_candidate_ids,
                             'reference':self.reference_ids,'backend':self.backend.contract_hash,
                             'validation_ids':self.validation_ids,'validation_labels':self.validation_labels})

    def select(self, *, snapshot, state, budget, spec):
        if budget != 32:
            raise ValueError('historical profiles require full B32 rounds')
        kwargs = {}
        if self.profile.acquisition == 'random':kwargs['original_candidate_ids']=self.original_candidate_ids
        if self.profile.acquisition == 'ivr':kwargs['reference_ids']=self.reference_ids
        if self.profile.acquisition == 'hybrid':
            kwargs['ensemble']=self.backend.ensemble(fit=state['fit'],state=state,
                validation_ids=self.validation_ids,validation_labels=self.validation_labels)
            kwargs['target_scales']=tuple(self.backend.preprocessing['target_scales'][k] for k in ('V1','V2'))
        result = select_numerical_profile(snapshot,profile=self.profile.name,outer_seed=spec.seed,
                                          round_index=state['round'],**kwargs).to_dict()
        result['audit']={**result['audit'],'profile':self.profile.name,'new_provider_calls':0}
        return result
