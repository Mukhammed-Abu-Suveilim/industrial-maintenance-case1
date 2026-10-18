# random_forest: результаты validation

Группа: `random_forest_20261017T203223_834932Z`.

- `failure_within_30d` / `A_numeric`: average_precision=0.778035; параметры поиска {"max_depth": 8, "min_samples_leaf": 5, "max_features": 1.0, "class_weight": null}.
- `failure_within_30d` / `A_tabular_restricted`: average_precision=0.776655; параметры поиска {"max_depth": 8, "min_samples_leaf": 5, "max_features": 1.0, "class_weight": null}.
- `fault_type` / `A_numeric`: macro_f1=0.770903; параметры поиска {"max_depth": 8, "min_samples_leaf": 5, "max_features": 1.0, "class_weight": null}.
- `fault_type` / `A_tabular_restricted`: macro_f1=0.772661; параметры поиска {"max_depth": 8, "min_samples_leaf": 5, "max_features": 1.0, "class_weight": null}.

## Пороги F2

- `A_numeric`: порог 0.18220070093700882; F2 0.571183 → 0.826603; precision 0.532925, recall 0.958678; FN 164 → 15, FP 91 → 305.
- `A_tabular_restricted`: порог 0.1872898061910771; F2 0.568312 → 0.826211; precision 0.532110, recall 0.958678; FN 165 → 15, FP 92 → 306.

## Полнота при ограничении precision

- `A_numeric`, минимум precision 70%: precision 0.705224, recall 0.520661; порог 0.5105102249890402.
- `A_numeric`, минимум precision 80%: precision 0.805263, recall 0.421488; порог 0.58707113337730998.
- `A_tabular_restricted`, минимум precision 70%: precision 0.700000, recall 0.539945; порог 0.50641425361932468.
- `A_tabular_restricted`, минимум precision 80%: precision 0.800000, recall 0.429752; порог 0.58299257091629608.

## Парные разности: candidate − reference

- vs_logistic_or_prior / `failure_within_30d` / `A_numeric`: Δaverage_precision=+0.091027; 95%-й интервал [+0.055762; +0.127923]; пригодных повторов 2000, пропущено 0; reference=6e7bc1cb575846deab4b1c55c23bd47f, candidate=6081e28363bd4d568e4ce486db5ac594.
- vs_logistic_or_prior / `failure_within_30d` / `A_numeric`: Δaverage_precision=+0.335842; 95%-й интервал [+0.289245; +0.381619]; пригодных повторов 2000, пропущено 0; reference=c0256cbf1a754a7d970c974b7923e481, candidate=6081e28363bd4d568e4ce486db5ac594.
- vs_logistic_or_prior / `failure_within_30d` / `A_numeric`: Δaverage_precision=+0.612509; 95%-й интервал [+0.544125; +0.673473]; пригодных повторов 2000, пропущено 0; reference=c2abf8dc3897473ab466615e5a546fb6, candidate=6081e28363bd4d568e4ce486db5ac594.
- vs_logistic_or_prior / `failure_within_30d` / `A_tabular_restricted`: Δaverage_precision=+0.093055; 95%-й интервал [+0.057832; +0.129879]; пригодных повторов 2000, пропущено 0; reference=f330e64b34a2445cbb93211953ac6542, candidate=3f231364cf124ad48ea72ee222ed53e6.
- vs_logistic_or_prior / `failure_within_30d` / `A_tabular_restricted`: Δaverage_precision=+0.334462; 95%-й интервал [+0.287897; +0.380985]; пригодных повторов 2000, пропущено 0; reference=c0256cbf1a754a7d970c974b7923e481, candidate=3f231364cf124ad48ea72ee222ed53e6.
- vs_logistic_or_prior / `failure_within_30d` / `A_tabular_restricted`: Δaverage_precision=+0.611129; 95%-й интервал [+0.540123; +0.671862]; пригодных повторов 2000, пропущено 0; reference=c2abf8dc3897473ab466615e5a546fb6, candidate=3f231364cf124ad48ea72ee222ed53e6.
- vs_logistic_or_prior / `fault_type` / `A_numeric`: Δmacro_f1=+0.000535; 95%-й интервал [-0.018682; +0.020765]; пригодных повторов 2000, пропущено 0; reference=c7a03ae5a8f142c3bb1fb392f0506fac, candidate=d08df3e322804da3b83bfb7059896b0e.
- vs_logistic_or_prior / `fault_type` / `A_numeric`: Δmacro_f1=-0.229097; 95%-й интервал [-0.265872; -0.213600]; пригодных повторов 2000, пропущено 0; reference=a524dbade83044728f1394cc1049ce1b, candidate=d08df3e322804da3b83bfb7059896b0e.
- vs_logistic_or_prior / `fault_type` / `A_numeric`: Δmacro_f1=+0.666150; 95%-й интервал [+0.613554; +0.705029]; пригодных повторов 2000, пропущено 0; reference=af0c6bda51b945378cd563fe9a46f245, candidate=d08df3e322804da3b83bfb7059896b0e.
- vs_logistic_or_prior / `fault_type` / `A_tabular_restricted`: Δmacro_f1=+0.006077; 95%-й интервал [-0.011600; +0.023162]; пригодных повторов 2000, пропущено 0; reference=71e68116dc904b408936c5df2ac04c71, candidate=41b88bbfca4142e48468fe70f9847ed5.
- vs_logistic_or_prior / `fault_type` / `A_tabular_restricted`: Δmacro_f1=-0.227339; 95%-й интервал [-0.264436; -0.210961]; пригодных повторов 2000, пропущено 0; reference=a524dbade83044728f1394cc1049ce1b, candidate=41b88bbfca4142e48468fe70f9847ed5.
- vs_logistic_or_prior / `fault_type` / `A_tabular_restricted`: Δmacro_f1=+0.667908; 95%-й интервал [+0.615260; +0.707370]; пригодных повторов 2000, пропущено 0; reference=af0c6bda51b945378cd563fe9a46f245, candidate=41b88bbfca4142e48468fe70f9847ed5.
- f2_vs_half / `failure_within_30d` / `A_numeric`: Δf2=+0.255421; 95%-й интервал [+0.210425; +0.310773]; пригодных повторов 2000, пропущено 0; reference=6081e28363bd4d568e4ce486db5ac594, candidate=6081e28363bd4d568e4ce486db5ac594.
- f2_vs_logistic / `failure_within_30d` / `A_numeric`: Δf2=+0.090505; 95%-й интервал [+0.061777; +0.121539]; пригодных повторов 2000, пропущено 0; reference=6e7bc1cb575846deab4b1c55c23bd47f, candidate=6081e28363bd4d568e4ce486db5ac594.
- f2_vs_half / `failure_within_30d` / `A_tabular_restricted`: Δf2=+0.257899; 95%-й интервал [+0.209919; +0.314004]; пригодных повторов 2000, пропущено 0; reference=3f231364cf124ad48ea72ee222ed53e6, candidate=3f231364cf124ad48ea72ee222ed53e6.
- f2_vs_logistic / `failure_within_30d` / `A_tabular_restricted`: Δf2=+0.085632; 95%-й интервал [+0.057610; +0.119560]; пригодных повторов 2000, пропущено 0; reference=f330e64b34a2445cbb93211953ac6542, candidate=3f231364cf124ad48ea72ee222ed53e6.

Конфигурации, F2-пороги и диагностические точки выбраны на одной validation. Оценки и интервалы условны относительно выбора и могут быть оптимистичными; итоговую тестовую оценку они не заменяют. Пороги в бутстрэпе фиксированы. Шесть общих текстовых шаблонов ограничивают перенос текстового контроля.

required_route, severity, id и обе цели исключены из входов. severity используется только в диагностике подгрупп. Test не оценивался. Нет объединения входов или прогнозов, калибровки и переобучения на train+validation.

Реализация этапа выполнена агентом по согласованному плану пользователя. Фактическое авторство остальных участников требует подтверждения.