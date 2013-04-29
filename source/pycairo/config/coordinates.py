from StringIO import StringIO
import numpy

# column names in WSS string below
coordinate_names = ['F', 'FDC', 'R', 'RX', 'RY', 'DX', 'DY', 'HX', 'HY', 'DHC', 'H']
# F - FPGA ID 0-11
# FDC - fpgaDncChannel 0-3
# R - ReticleID in wafer production map
# RX - X-coordinate of reticle in wafer production map
# RX - Y-coordinate of reticle in wafer production map
# DX - X-coordinate of dnc(reticle) in cartesian coordinates (halbe coordinates)
# DY - Y-coordinate of dnc(reticle) in cartesian coordinates (halbe coordinates)
# HX - X-coordinate of HICANN in cartesian coordinates (halbe coordinates)
# HY - Y-coordinate of HICANN in cartesian coordinates (halbe coordinates)
# DHC - dncHicannChannel 0-7
# H - Hicann config Id (halbe coordinates)
#
# F FDC R RX RY DX DY HX HY DHC H
WSS = """0 0 21 5 3 4 5 19 10 6 279
0 0 21 5 3 4 5 19 11 7 307
0 0 21 5 3 4 5 18 10 4 278
0 0 21 5 3 4 5 18 11 5 306
0 0 21 5 3 4 5 17 10 2 277
0 0 21 5 3 4 5 17 11 3 305
0 0 21 5 3 4 5 16 10 0 276
0 0 21 5 3 4 5 16 11 1 304
0 1 22 6 3 5 5 23 10 6 283
0 1 22 6 3 5 5 23 11 7 311
0 1 22 6 3 5 5 22 10 4 282
0 1 22 6 3 5 5 22 11 5 310
0 1 22 6 3 5 5 21 10 2 281
0 1 22 6 3 5 5 21 11 3 309
0 1 22 6 3 5 5 20 10 0 280
0 1 22 6 3 5 5 20 11 1 308
0 2 23 7 4 6 4 27 8 6 219
0 2 23 7 4 6 4 27 9 7 255
0 2 23 7 4 6 4 26 8 4 218
0 2 23 7 4 6 4 26 9 5 254
0 2 23 7 4 6 4 25 8 2 217
0 2 23 7 4 6 4 25 9 3 253
0 2 23 7 4 6 4 24 8 0 216
0 2 23 7 4 6 4 24 9 1 252
0 3 24 6 4 5 4 23 8 6 215
0 3 24 6 4 5 4 23 9 7 251
0 3 24 6 4 5 4 22 8 4 214
0 3 24 6 4 5 4 22 9 5 250
0 3 24 6 4 5 4 21 8 2 213
0 3 24 6 4 5 4 21 9 3 249
0 3 24 6 4 5 4 20 8 0 212
0 3 24 6 4 5 4 20 9 1 248
1 0 9 7 6 6 2 27 4 6 87
1 0 9 7 6 6 2 27 5 7 115
1 0 9 7 6 6 2 26 4 4 86
1 0 9 7 6 6 2 26 5 5 114
1 0 9 7 6 6 2 25 4 2 85
1 0 9 7 6 6 2 25 5 3 113
1 0 9 7 6 6 2 24 4 0 84
1 0 9 7 6 6 2 24 5 1 112
1 1 10 6 6 5 2 23 4 6 83
1 1 10 6 6 5 2 23 5 7 111
1 1 10 6 6 5 2 22 4 4 82
1 1 10 6 6 5 2 22 5 5 110
1 1 10 6 6 5 2 21 4 2 81
1 1 10 6 6 5 2 21 5 3 109
1 1 10 6 6 5 2 20 4 0 80
1 1 10 6 6 5 2 20 5 1 108
1 2 11 5 6 4 2 19 4 6 79
1 2 11 5 6 4 2 19 5 7 107
1 2 11 5 6 4 2 18 4 4 78
1 2 11 5 6 4 2 18 5 5 106
1 2 11 5 6 4 2 17 4 2 77
1 2 11 5 6 4 2 17 5 3 105
1 2 11 5 6 4 2 16 4 0 76
1 2 11 5 6 4 2 16 5 1 104
1 3 12 6 5 5 3 23 6 6 143
1 3 12 6 5 5 3 23 7 7 179
1 3 12 6 5 5 3 22 6 4 142
1 3 12 6 5 5 3 22 7 5 178
1 3 12 6 5 5 3 21 6 2 141
1 3 12 6 5 5 3 21 7 3 177
1 3 12 6 5 5 3 20 6 0 140
1 3 12 6 5 5 3 20 7 1 176
2 0 45 4 7 3 1 15 2 6 31
2 0 45 4 7 3 1 15 3 7 51
2 0 45 4 7 3 1 14 2 4 30
2 0 45 4 7 3 1 14 3 5 50
2 0 45 4 7 3 1 13 2 2 29
2 0 45 4 7 3 1 13 3 3 49
2 0 45 4 7 3 1 12 2 0 28
2 0 45 4 7 3 1 12 3 1 48
2 1 46 5 5 4 3 19 6 6 139
2 1 46 5 5 4 3 19 7 7 175
2 1 46 5 5 4 3 18 6 4 138
2 1 46 5 5 4 3 18 7 5 174
2 1 46 5 5 4 3 17 6 2 137
2 1 46 5 5 4 3 17 7 3 173
2 1 46 5 5 4 3 16 6 0 136
2 1 46 5 5 4 3 16 7 1 172
2 2 47 3 5 2 3 11 6 6 131
2 2 47 3 5 2 3 11 7 7 167
2 2 47 3 5 2 3 10 6 4 130
2 2 47 3 5 2 3 10 7 5 166
2 2 47 3 5 2 3 9 6 2 129
2 2 47 3 5 2 3 9 7 3 165
2 2 47 3 5 2 3 8 6 0 128
2 2 47 3 5 2 3 8 7 1 164
2 3 48 4 5 3 3 15 6 6 135
2 3 48 4 5 3 3 15 7 7 171
2 3 48 4 5 3 3 14 6 4 134
2 3 48 4 5 3 3 14 7 5 170
2 3 48 4 5 3 3 13 6 2 133
2 3 48 4 5 3 3 13 7 3 169
2 3 48 4 5 3 3 12 6 0 132
2 3 48 4 5 3 3 12 7 1 168
3 0 33 3 3 2 5 11 10 6 271
3 0 33 3 3 2 5 11 11 7 299
3 0 33 3 3 2 5 10 10 4 270
3 0 33 3 3 2 5 10 11 5 298
3 0 33 3 3 2 5 9 10 2 269
3 0 33 3 3 2 5 9 11 3 297
3 0 33 3 3 2 5 8 10 0 268
3 0 33 3 3 2 5 8 11 7 296
3 1 34 5 4 4 4 19 8 6 211
3 1 34 5 4 4 4 19 9 7 247
3 1 34 5 4 4 4 18 8 4 210
3 1 34 5 4 4 4 18 9 5 246
3 1 34 5 4 4 4 17 8 2 209
3 1 34 5 4 4 4 17 9 3 245
3 1 34 5 4 4 4 16 8 0 208
3 1 34 5 4 4 4 16 9 1 244
3 2 35 4 3 3 5 15 10 6 275
3 2 35 4 3 3 5 15 11 7 303
3 2 35 4 3 3 5 14 10 4 274
3 2 35 4 3 3 5 14 11 5 302
3 2 35 4 3 3 5 13 10 2 273
3 2 35 4 3 3 5 13 11 3 301
3 2 35 4 3 3 5 12 10 0 272
3 2 35 4 3 3 5 12 11 1 300
3 3 36 4 4 3 4 15 8 6 207
3 3 36 4 4 3 4 15 9 7 243
3 3 36 4 4 3 4 14 8 4 206
3 3 36 4 4 3 4 14 9 5 242
3 3 36 4 4 3 4 13 8 2 205
3 3 36 4 4 3 4 13 9 3 241
3 3 36 4 4 3 4 12 8 0 204
3 3 36 4 4 3 4 12 9 1 240
4 0 41 1 5 0 3 3 6 6 123
4 0 41 1 5 0 3 3 7 7 159
4 0 41 1 5 0 3 2 6 4 122
4 0 41 1 5 0 3 2 7 5 158
4 0 41 1 5 0 3 1 6 2 121
4 0 41 1 5 0 3 1 7 3 157
4 0 41 1 5 0 3 0 6 0 120
4 0 41 1 5 0 3 0 7 1 156
4 1 42 2 5 1 3 7 6 6 127
4 1 42 2 5 1 3 7 7 7 163
4 1 42 2 5 1 3 6 6 4 126
4 1 42 2 5 1 3 6 7 5 162
4 1 42 2 5 1 3 5 6 2 125
4 1 42 2 5 1 3 5 7 3 161
4 1 42 2 5 1 3 4 6 0 124
4 1 42 2 5 1 3 4 7 1 160
4 2 43 3 6 2 2 11 4 6 71
4 2 43 3 6 2 2 11 5 7 99
4 2 43 3 6 2 2 10 4 4 70
4 2 43 3 6 2 2 10 5 5 98
4 2 43 3 6 2 2 9 4 2 69
4 2 43 3 6 2 2 9 5 3 97
4 2 43 3 6 2 2 8 4 0 68
4 2 43 3 6 2 2 8 5 1 96
4 3 44 2 6 1 2 7 4 6 67
4 3 44 2 6 1 2 7 5 7 95
4 3 44 2 6 1 2 6 4 4 66
4 3 44 2 6 1 2 6 5 5 94
4 3 44 2 6 1 2 5 4 2 65
4 3 44 2 6 1 2 5 5 3 93
4 3 44 2 6 1 2 4 4 0 64
4 3 44 2 6 1 2 4 5 1 92
5 0 1 4 8 3 0 15 0 6 3
5 0 1 4 8 3 0 15 1 7 15
5 0 1 4 8 3 0 14 0 4 2
5 0 1 4 8 3 0 14 1 5 14
5 0 1 4 8 3 0 13 0 2 1
5 0 1 4 8 3 0 13 1 3 13
5 0 1 4 8 3 0 12 0 0 0
5 0 1 4 8 3 0 12 1 1 12
5 1 2 3 7 2 1 11 2 6 27
5 1 2 3 7 2 1 11 3 7 47
5 1 2 3 7 2 1 10 2 4 26
5 1 2 3 7 2 1 10 3 5 46
5 1 2 3 7 2 1 9 2 2 25
5 1 2 3 7 2 1 9 3 3 45
5 1 2 3 7 2 1 8 2 0 24
5 1 2 3 7 2 1 8 3 1 44
5 2 3 4 6 3 2 15 4 6 75
5 2 3 4 6 3 2 15 5 7 103
5 2 3 4 6 3 2 14 4 4 74
5 2 3 4 6 3 2 14 5 5 102
5 2 3 4 6 3 2 13 4 2 73
5 2 3 4 6 3 2 13 5 3 101
5 2 3 4 6 3 2 12 4 0 72
5 2 3 4 6 3 2 12 5 1 100
5 3 4 5 8 4 0 19 0 6 7
5 3 4 5 8 4 0 19 1 7 19
5 3 4 5 8 4 0 18 0 4 6
5 3 4 5 8 4 0 18 1 5 18
5 3 4 5 8 4 0 17 0 2 5
5 3 4 5 8 4 0 17 1 3 17
5 3 4 5 8 4 0 16 0 0 4
5 3 4 5 8 4 0 16 1 1 16
6 0 5 6 8 5 0 23 0 6 11
6 0 5 6 8 5 0 23 1 7 23
6 0 5 6 8 5 0 22 0 4 10
6 0 5 6 8 5 0 22 1 5 22
6 0 5 6 8 5 0 21 0 2 9
6 0 5 6 8 5 0 21 1 3 21
6 0 5 6 8 5 0 20 0 0 8
6 0 5 6 8 5 0 20 1 1 20
6 1 6 5 7 4 1 19 2 6 35
6 1 6 5 7 4 1 19 3 7 55
6 1 6 5 7 4 1 18 2 4 34
6 1 6 5 7 4 1 18 3 5 54
6 1 6 5 7 4 1 17 2 2 33
6 1 6 5 7 4 1 17 3 3 53
6 1 6 5 7 4 1 16 2 0 32
6 1 6 5 7 4 1 16 3 1 52
6 2 7 6 7 5 1 23 2 6 39
6 2 7 6 7 5 1 23 3 7 59
6 2 7 6 7 5 1 22 2 4 38
6 2 7 6 7 5 1 22 3 5 58
6 2 7 6 7 5 1 21 2 2 37
6 2 7 6 7 5 1 21 3 3 57
6 2 7 6 7 5 1 20 2 0 36
6 2 7 6 7 5 1 20 3 1 56
6 3 8 7 7 6 1 27 2 6 43
6 3 8 7 7 6 1 27 3 7 63
6 3 8 7 7 6 1 26 2 4 42
6 3 8 7 7 6 1 26 3 5 62
6 3 8 7 7 6 1 25 2 2 41
6 3 8 7 7 6 1 25 3 3 61
6 3 8 7 7 6 1 24 2 0 40
6 3 8 7 7 6 1 24 3 1 60
7 0 13 9 5 8  3 35 6 6 155
7 0 13 9 5 8  3 35 7 7 191
7 0 13 9 5 8  3 34 6 4 154
7 0 13 9 5 8  3 34 7 5 190
7 0 13 9 5 8  3 33 6 2 153
7 0 13 9 5 8  3 33 7 3 189
7 0 13 9 5 8  3 32 6 0 152
7 0 13 9 5 8  3 32 7 1 188
7 1 14 8 6 7 2 31 4 6 91
7 1 14 8 6 7 2 31 5 7 119
7 1 14 8 6 7 2 30 4 4 90
7 1 14 8 6 7 2 30 5 5 118
7 1 14 8 6 7 2 29 4 2 89
7 1 14 8 6 7 2 29 5 3 117
7 1 14 8 6 7 2 28 4 0 88
7 1 14 8 6 7 2 28 5 1 116
7 2 15 7 5 6 3 27 6 6 147
7 2 15 7 5 6 3 27 7 7 183
7 2 15 7 5 6 3 26 6 4 146
7 2 15 7 5 6 3 26 7 5 182
7 2 15 7 5 6 3 25 6 2 145
7 2 15 7 5 6 3 25 7 3 181
7 2 15 7 5 6 3 24 6 0 144
7 2 15 7 5 6 3 24 7 1 180
7 3 16 8 5 7 3 31 6 6 151
7 3 16 8 5 7 3 31 7 7 187
7 3 16 8 5 7 3 30 6 4 150
7 3 16 8 5 7 3 30 7 5 186
7 3 16 8 5 7 3 29 6 2 149
7 3 16 8 5 7 3 29 7 3 185
7 3 16 8 5 7 3 28 6 0 148
7 3 16 8 5 7 3 28 7 1 184
8 0 17 9 4 8  4 35 8 6 227
8 0 17 9 4 8  4 35 9 7 263
8 0 17 9 4 8  4 34 8 4 226
8 0 17 9 4 8  4 34 9 5 262
8 0 17 9 4 8  4 33 8 2 225
8 0 17 9 4 8  4 33 9 3 261
8 0 17 9 4 8  4 32 8 0 224
8 0 17 9 4 8  4 32 9 1 260
8 1 18 8 4 7 4 31 8 6 223
8 1 18 8 4 7 4 31 9 7 259
8 1 18 8 4 7 4 30 8 4 222
8 1 18 8 4 7 4 30 9 5 258
8 1 18 8 4 7 4 29 8 2 221
8 1 18 8 4 7 4 29 9 3 257
8 1 18 8 4 7 4 28 8 0 220
8 1 18 8 4 7 4 28 9 1 256
8 2 19 7 3 6 5 27 10 6 287
8 2 19 7 3 6 5 27 11 7 315
8 2 19 7 3 6 5 26 10 4 286
8 2 19 7 3 6 5 26 11 5 314
8 2 19 7 3 6 5 25 10 2 285
8 2 19 7 3 6 5 25 11 3 313
8 2 19 7 3 6 5 24 10 0 284
8 2 19 7 3 6 5 24 11 7 312
8 3 20 8 3 7 5 31 10 6 291
8 3 20 8 3 7 5 31 11 7 319
8 3 20 8 3 7 5 30 10 4 290
8 3 20 8 3 7 5 30 11 5 318
8 3 20 8 3 7 5 29 10 2 289
8 3 20 8 3 7 5 29 11 3 317
8 3 20 8 3 7 5 28 10 0 288
8 3 20 8 3 7 5 28 11 1 316
9 0 25 6 1 5 7 23 14 6 371
9 0 25 6 1 5 7 23 15 7 383
9 0 25 6 1 5 7 22 14 4 370
9 0 25 6 1 5 7 22 15 5 382
9 0 25 6 1 5 7 21 14 2 369
9 0 25 6 1 5 7 21 15 3 381
9 0 25 6 1 5 7 20 14 0 368
9 0 25 6 1 5 7 20 15 1 380
9 1 26 7 2 6 6 27 12 6 339
9 1 26 7 2 6 6 27 13 7 359
9 1 26 7 2 6 6 26 12 4 338
9 1 26 7 2 6 6 26 13 5 358
9 1 26 7 2 6 6 25 12 2 337
9 1 26 7 2 6 6 25 13 3 357
9 1 26 7 2 6 6 24 12 0 336
9 1 26 7 2 6 6 24 13 1 356
9 2 27 5 2 4 6 19 12 6 331
9 2 27 5 2 4 6 19 13 7 351
9 2 27 5 2 4 6 18 12 4 330
9 2 27 5 2 4 6 18 13 5 350
9 2 27 5 2 4 6 17 12 2 329
9 2 27 5 2 4 6 17 13 3 349
9 2 27 5 2 4 6 16 12 0 328
9 2 27 5 2 4 6 16 13 1 348
9 3 28 6 2 5 6 23 12 6 335
9 3 28 6 2 5 6 23 13 7 355
9 3 28 6 2 5 6 22 12 4 334
9 3 28 6 2 5 6 22 13 5 354
9 3 28 6 2 5 6 21 12 2 333
9 3 28 6 2 5 6 21 13 3 353
9 3 28 6 2 5 6 20 12 0 332
9 3 28 6 2 5 6 20 13 1 352
10 0 29 4 1 3 7 15 14 6 363
10 0 29 4 1 3 7 15 15 7 375
10 0 29 4 1 3 7 14 14 4 362
10 0 29 4 1 3 7 14 15 5 374
10 0 29 4 1 3 7 13 14 2 361
10 0 29 4 1 3 7 13 15 3 373
10 0 29 4 1 3 7 12 14 0 360
10 0 29 4 1 3 7 12 15 1 372
10 1 30 5 1 4 7 19 14 6 367
10 1 30 5 1 4 7 19 15 7 379
10 1 30 5 1 4 7 18 14 4 366
10 1 30 5 1 4 7 18 15 5 378
10 1 30 5 1 4 7 17 14 2 365
10 1 30 5 1 4 7 17 15 3 377
10 1 30 5 1 4 7 16 14 0 364
10 1 30 5 1 4 7 16 15 1 376
10 2 31 4 2 3 6 15 12 6 327
10 2 31 4 2 3 6 15 13 7 347
10 2 31 4 2 3 6 14 12 4 326
10 2 31 4 2 3 6 14 13 5 346
10 2 31 4 2 3 6 13 12 2 325
10 2 31 4 2 3 6 13 13 3 345
10 2 31 4 2 3 6 12 12 0 324
10 2 31 4 2 3 6 12 13 1 344
10 3 32 3 2 2 6 11 12 6 323
10 3 32 3 2 2 6 11 13 7 343
10 3 32 3 2 2 6 10 12 4 322
10 3 32 3 2 2 6 10 13 5 342
10 3 32 3 2 2 6 9 12 2 321
10 3 32 3 2 2 6 9 13 3 341
10 3 32 3 2 2 6 8 12 0 320
10 3 32 3 2 2 6 8 13 1 340
11 0 37 1 4 0 4 3 8 6 195
11 0 37 1 4 0 4 3 9 7 231
11 0 37 1 4 0 4 2 8 4 194
11 0 37 1 4 0 4 2 9 5 230
11 0 37 1 4 0 4 1 8 2 193
11 0 37 1 4 0 4 1 9 3 229
11 0 37 1 4 0 4 0 8 0 192
11 0 37 1 4 0 4 0 9 1 228
11 1 38 2 3 1 5 7 10 6 267
11 1 38 2 3 1 5 7 11 7 295
11 1 38 2 3 1 5 6 10 4 266
11 1 38 2 3 1 5 6 11 5 294
11 1 38 2 3 1 5 5 10 2 265
11 1 38 2 3 1 5 5 11 3 293
11 1 38 2 3 1 5 4 10 0 264
11 1 38 2 3 1 5 4 11 1 292
11 2 39 3 4 2 4 11 8 6 203
11 2 39 3 4 2 4 11 9 7 239
11 2 39 3 4 2 4 10 8 4 202
11 2 39 3 4 2 4 10 9 5 238
11 2 39 3 4 2 4 9 8 2 201
11 2 39 3 4 2 4 9 9 3 237
11 2 39 3 4 2 4 8 8 0 200
11 2 39 3 4 2 4 8 9 1 236
11 3 40 2 4 1 4 7 8 6 199
11 3 40 2 4 1 4 7 9 7 235
11 3 40 2 4 1 4 6 8 4 198
11 3 40 2 4 1 4 6 9 5 234
11 3 40 2 4 1 4 5 8 2 197
11 3 40 2 4 1 4 5 9 3 233
11 3 40 2 4 1 4 4 8 0 196
11 3 40 2 4 1 4 4 9 1 232"""

FPGA = """0 S 2 4
1 E 4 2
2 N 2 0
3 W 0 2
4 SSW 1 4
5 SSE 3 4
6 ESE 4 3
7 ENE 4 1
8 NNE 3 0
9 NNW 1 0
10 WNW 0 1
11 WSW 0 3"""

IP = """0 1
1 3
2 5
3 7
4 9
5 11
6 13
7 15
8 17
9 19
10 21
11 23"""

        
def get_reticle_map():
    reticles = numpy.genfromtxt(StringIO(WSS), names=coordinate_names, dtype=int)
    reticles.sort(order=['F', 'FDC', 'DHC']) # Sort by FPGA ID, fpga-dnc-channel, dnc-hicann-channel

    # reshape this ndarray, such that there is a quick access via
    # reticle_map[f][fdc][dhc]
    reticle_map = numpy.reshape(reticles, (12,4,8))
    return reticle_map

def get_fpga_map():
    fpga_coordinates = numpy.genfromtxt(StringIO(FPGA))

    fpga_map = []
    for i in fpga_coordinates:
        fpga_map.append(i)

    return fpga_map
        
def get_fpga_ip():
    fpga_ip_file = numpy.genfromtxt(StringIO(IP))

    fpga_ip = []
    for i in fpga_ip_file:
        fpga_ip.append(i)

    return fpga_ip
