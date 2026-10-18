# catboost: результаты validation

Группа: `catboost_20261017T203913_936394Z`.

- `failure_within_30d` / `A_numeric`: average_precision=0.786742; параметры поиска {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10, "auto_class_weights": null}.
- `failure_within_30d` / `A_tabular_restricted`: average_precision=0.783998; параметры поиска {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10, "auto_class_weights": null}.
- `fault_type` / `A_numeric`: macro_f1=0.757161; параметры поиска {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10, "auto_class_weights": null}.
- `fault_type` / `A_tabular_restricted`: macro_f1=0.773016; параметры поиска {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 10, "auto_class_weights": null}.

## Пороги F2

- `A_numeric`: порог 0.13918557568898432; F2 0.627828 → 0.827506; precision 0.512266, recall 0.977961; FN 141 → 8, FP 94 → 338.
- `A_tabular_restricted`: порог 0.13651169820466486; F2 0.602273 → 0.824048; precision 0.505698, recall 0.977961; FN 151 → 8, FP 96 → 347.

## Полнота при ограничении precision

- `A_numeric`, минимум precision 70%: precision 0.704403, recall 0.617080; порог 0.49828994617946826.
- `A_numeric`, минимум precision 80%: precision 0.815000, recall 0.449036; порог 0.63621663408742035.
- `A_tabular_restricted`, минимум precision 70%: precision 0.700348, recall 0.553719; порог 0.52197315287866097.
- `A_tabular_restricted`, минимум precision 80%: precision 0.801802, recall 0.490358; порог 0.59982034523712358.

## Парные разности: candidate − reference

- vs_logistic_or_prior / `failure_within_30d` / `A_numeric`: Δaverage_precision=+0.099733; 95%-й интервал [+0.065944; +0.133300]; пригодных повторов 2000, пропущено 0; reference=6e7bc1cb575846deab4b1c55c23bd47f, candidate=b1169d15559841ca8d83f48cb2c4ff87.
- vs_logistic_or_prior / `failure_within_30d` / `A_numeric`: Δaverage_precision=+0.344549; 95%-й интервал [+0.301533; +0.388192]; пригодных повторов 2000, пропущено 0; reference=c0256cbf1a754a7d970c974b7923e481, candidate=b1169d15559841ca8d83f48cb2c4ff87.
- vs_logistic_or_prior / `failure_within_30d` / `A_numeric`: Δaverage_precision=+0.621215; 95%-й интервал [+0.552201; +0.679005]; пригодных повторов 2000, пропущено 0; reference=c2abf8dc3897473ab466615e5a546fb6, candidate=b1169d15559841ca8d83f48cb2c4ff87.
- catboost_vs_forest / `failure_within_30d` / `A_numeric`: Δaverage_precision=+0.008707; 95%-й интервал [-0.003035; +0.020497]; пригодных повторов 2000, пропущено 0; reference=6081e28363bd4d568e4ce486db5ac594, candidate=b1169d15559841ca8d83f48cb2c4ff87.
- vs_logistic_or_prior / `failure_within_30d` / `A_tabular_restricted`: Δaverage_precision=+0.100398; 95%-й интервал [+0.066565; +0.133447]; пригодных повторов 2000, пропущено 0; reference=f330e64b34a2445cbb93211953ac6542, candidate=0ab144d540c0465c8f663c6e6872501f.
- vs_logistic_or_prior / `failure_within_30d` / `A_tabular_restricted`: Δaverage_precision=+0.341805; 95%-й интервал [+0.296083; +0.386517]; пригодных повторов 2000, пропущено 0; reference=c0256cbf1a754a7d970c974b7923e481, candidate=0ab144d540c0465c8f663c6e6872501f.
- vs_logistic_or_prior / `failure_within_30d` / `A_tabular_restricted`: Δaverage_precision=+0.618472; 95%-й интервал [+0.549262; +0.677577]; пригодных повторов 2000, пропущено 0; reference=c2abf8dc3897473ab466615e5a546fb6, candidate=0ab144d540c0465c8f663c6e6872501f.
- catboost_vs_forest / `failure_within_30d` / `A_tabular_restricted`: Δaverage_precision=+0.007343; 95%-й интервал [-0.006299; +0.021271]; пригодных повторов 2000, пропущено 0; reference=3f231364cf124ad48ea72ee222ed53e6, candidate=0ab144d540c0465c8f663c6e6872501f.
- vs_logistic_or_prior / `fault_type` / `A_numeric`: Δmacro_f1=-0.013207; 95%-й интервал [-0.034742; +0.008034]; пригодных повторов 2000, пропущено 0; reference=c7a03ae5a8f142c3bb1fb392f0506fac, candidate=3b50968515fe4bfbaa3c44efd65aa670.
- vs_logistic_or_prior / `fault_type` / `A_numeric`: Δmacro_f1=-0.242839; 95%-й интервал [-0.279652; -0.222900]; пригодных повторов 2000, пропущено 0; reference=a524dbade83044728f1394cc1049ce1b, candidate=3b50968515fe4bfbaa3c44efd65aa670.
- vs_logistic_or_prior / `fault_type` / `A_numeric`: Δmacro_f1=+0.652408; 95%-й интервал [+0.600556; +0.694444]; пригодных повторов 2000, пропущено 0; reference=af0c6bda51b945378cd563fe9a46f245, candidate=3b50968515fe4bfbaa3c44efd65aa670.
- catboost_vs_forest / `fault_type` / `A_numeric`: Δmacro_f1=-0.013742; 95%-й интервал [-0.032286; +0.003984]; пригодных повторов 2000, пропущено 0; reference=d08df3e322804da3b83bfb7059896b0e, candidate=3b50968515fe4bfbaa3c44efd65aa670.
- vs_logistic_or_prior / `fault_type` / `A_tabular_restricted`: Δmacro_f1=+0.006432; 95%-й интервал [-0.009023; +0.021796]; пригодных повторов 2000, пропущено 0; reference=71e68116dc904b408936c5df2ac04c71, candidate=832874e1ae454ca897b6ea2a34d6f327.
- vs_logistic_or_prior / `fault_type` / `A_tabular_restricted`: Δmacro_f1=-0.226984; 95%-й интервал [-0.261938; -0.211221]; пригодных повторов 2000, пропущено 0; reference=a524dbade83044728f1394cc1049ce1b, candidate=832874e1ae454ca897b6ea2a34d6f327.
- vs_logistic_or_prior / `fault_type` / `A_tabular_restricted`: Δmacro_f1=+0.668263; 95%-й интервал [+0.617389; +0.707908]; пригодных повторов 2000, пропущено 0; reference=af0c6bda51b945378cd563fe9a46f245, candidate=832874e1ae454ca897b6ea2a34d6f327.
- catboost_vs_forest / `fault_type` / `A_tabular_restricted`: Δmacro_f1=+0.000355; 95%-й интервал [-0.013450; +0.013887]; пригодных повторов 2000, пропущено 0; reference=41b88bbfca4142e48468fe70f9847ed5, candidate=832874e1ae454ca897b6ea2a34d6f327.
- f2_vs_half / `failure_within_30d` / `A_numeric`: Δf2=+0.199678; 95%-й интервал [+0.151635; +0.254906]; пригодных повторов 2000, пропущено 0; reference=b1169d15559841ca8d83f48cb2c4ff87, candidate=b1169d15559841ca8d83f48cb2c4ff87.
- f2_vs_logistic / `failure_within_30d` / `A_numeric`: Δf2=+0.091407; 95%-й интервал [+0.064037; +0.123502]; пригодных повторов 2000, пропущено 0; reference=6e7bc1cb575846deab4b1c55c23bd47f, candidate=b1169d15559841ca8d83f48cb2c4ff87.
- f2_vs_half / `failure_within_30d` / `A_tabular_restricted`: Δf2=+0.221776; 95%-й интервал [+0.173725; +0.276816]; пригодных повторов 2000, пропущено 0; reference=0ab144d540c0465c8f663c6e6872501f, candidate=0ab144d540c0465c8f663c6e6872501f.
- f2_vs_logistic / `failure_within_30d` / `A_tabular_restricted`: Δf2=+0.083470; 95%-й интервал [+0.056600; +0.117092]; пригодных повторов 2000, пропущено 0; reference=f330e64b34a2445cbb93211953ac6542, candidate=0ab144d540c0465c8f663c6e6872501f.

Конфигурации, F2-пороги и диагностические точки выбраны на одной validation. Оценки и интервалы условны относительно выбора и могут быть оптимистичными; итоговую тестовую оценку они не заменяют. Пороги в бутстрэпе фиксированы. Шесть общих текстовых шаблонов ограничивают перенос текстового контроля.

required_route, severity, id и обе цели исключены из входов. severity используется только в диагностике подгрупп. Test не оценивался. Нет объединения входов или прогнозов, калибровки и переобучения на train+validation.

Реализация этапа выполнена агентом по согласованному плану пользователя. Фактическое авторство остальных участников требует подтверждения.