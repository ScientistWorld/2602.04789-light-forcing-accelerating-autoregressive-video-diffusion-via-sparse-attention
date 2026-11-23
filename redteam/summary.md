# Red-team rank approximation (positive = outranks the reference)

| folder | file | method | core err / active | long err / active | raw core | raw long | raw total | z total | verdict (raw / z) |
|---|---|---|---|---|---|---|---|---|---|
| (gym scores_test.json) | - | light_forcing | 0.472 / 0.104 | 0.485 / 0.104 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ties / ties |
| (gym scores_test.json) | - | light_forcing_uniform | 0.493 / 0.097 | 0.490 / 0.100 | -0.0222 | 0.0255 | -0.0011 | -0.2554 | below ref / below ref |
| (gym scores_test.json) | - | one_d_sparse | 0.493 / 0.131 | 0.490 / 0.120 | -0.0390 | 0.0153 | -0.0122 | -0.3353 | below ref / below ref |
| (gym scores_test.json) | - | sliding_window | 0.349 / 0.464 | 0.544 / 0.378 | 0.1465 | -0.5885 | -0.1400 | -0.4366 | below ref / below ref |
