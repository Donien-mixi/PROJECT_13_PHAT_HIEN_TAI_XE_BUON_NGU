# ðŸ§¬ BÃ¡o cÃ¡o Build Clean Dataset (Äá»“ Ã¡n 13)

- **Thá»i gian build:** 907.5s
- **File npz:** `preprocessed_driver_dataset.npz`
- **Tá»•ng máº«u há»£p lá»‡:** 7651 (train 6894 / val giá»¯-out 757)
- **Preview kiá»ƒm tra:** D:\PROJECT_13_PHAT_HIEN_BUON_NGU\output\preprocessed_preview

## ðŸ“¥ Sá»‘ máº«u theo nguá»“n

| Nguá»“n | Há»£p lá»‡ | Train | Val |
|---|---|---|---|
| AFLW2000_3D | 1622 | 1494 | 128 |
| CEW | 2450 | 2255 | 195 |
| YawDD | 3579 | 3145 | 434 |

## ðŸ—‘ï¸ Máº«u bá»‹ loáº¡i theo lÃ½ do (QA Gates)

| Nguá»“n | LÃ½ do | Sá»‘ máº«u |
|---|---|---|
| AFLW2000_3D | teacher_no_face | 245 |
| AFLW2000_3D | duplicate | 65 |
| AFLW2000_3D | extreme_pose | 39 |
| AFLW2000_3D | teacher_native_mismatch | 21 |
| AFLW2000_3D | face_too_small | 7 |
| AFLW2000_3D | image_missing | 3 |
| AFLW2000_3D | landmark_clipped | 1 |
| CEW | duplicate | 1850 |
| CEW | teacher_no_face | 174 |
| CEW | extreme_pose | 58 |
| YawDD | duplicate | 7113 |
| YawDD | extreme_pose | 37 |
| YawDD | face_too_small | 1 |

## ðŸ“ PhÃ¢n bá»‘ gÃ³c Yaw (Ä‘á»™) â€” pháº£i phá»§ Ä‘á»u 2 phÃ­a

| Khoáº£ng | Sá»‘ máº«u |
|---|---|
| [-90, -40) | 244 |
| [-40, -20) | 737 |
| [-20, +20) | 4127 |
| [+20, +40) | 1747 |
| [+40, +90) | 796 |

## ðŸ¥± PhÃ¢n bá»‘ tráº¡ng thÃ¡i sinh tráº¯c

| Tráº¡ng thÃ¡i | NgÆ°á»¡ng | Sá»‘ máº«u |
|---|---|---|
| NgÃ¡p (MAR cao) | MAR >= 0.40 | 1920 |
| Nháº¯m máº¯t | EAR < 0.21 | 3347 |
| BÃ¬nh thÆ°á»ng | cÃ²n láº¡i | 3077 |

> âš ï¸ Náº¿u cá»™t Yaw [-90,-40) hoáº·c [40,90) báº±ng 0 â†’ bá»• sung 300W-LP/AFLW2000
Ä‘á»ƒ mÃ´ hÃ¬nh há»c báº¥t biáº¿n gÃ³c quay (nguyÃªn nhÃ¢n tracking há»ng cÅ©).