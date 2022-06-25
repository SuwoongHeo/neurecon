# CIHP Definition
CLASSES = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
               'dress', 'coat', 'socks', 'pants', 'tosor-skin', 'scarf', 'skirt',
               'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe', 'rightShoe']

# Note, it is in BGR order
CIHP20 = {0: [0, 0, 0], 1: [0, 0, 128], 2: [0, 0, 255], 3: [0, 85, 0], 4: [51, 0, 170], 5: [0, 85, 255],
          6: [85, 0, 0], 7: [221, 119, 0], 8: [0, 85, 85], 9: [85, 85, 0], 10: [0, 51, 85],
          11: [128, 86, 52], 12: [0, 128, 0], 13: [255, 0, 0], 14: [221, 170, 51], 15: [255, 255, 0],
          16: [170, 255, 85], 17: [85, 255, 170], 18: [0, 255, 255], 19: [0, 170, 255]}

# Naked Body, Upper , Lower, Background setup
# 1 (Naked Body) : 2 (Hair), 10 (Torso-Skin), 13 (Face), 14, 15(Left, Right Arm), 16, 17(Left, Right Leg)
# 2 (Upper Garments) : 1 (Hat), 3 (Glove), 4 (Sunglasses), 5 (Upper-Clothes), 6 (Dress), 7 (Coat), 11 (Scarf)
# 3 (Lower Garments) : 8 (Socks), 9 (Patns), 12 (Skirt), 18 (LeftShoe), 19 (RightShoe)
# 0 (Background) : 0 (Background)
NULB_CMAP = {0: [0], 1: [2, 10, 13, 14, 15, 16, 17], 2: [1, 3, 4, 5, 6, 7, 11] , 3: [8, 9, 12, 18, 19]}
