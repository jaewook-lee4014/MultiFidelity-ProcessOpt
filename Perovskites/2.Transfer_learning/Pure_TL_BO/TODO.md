# TODO List

## DNGO 성능 향상 실험 (우선순위 순)
- [ ] 1. 차별적 학습률 (Discriminative LR) 적용
      - Head에 높은 LR, backbone에 낮은 LR
- [x] 2. 점진적 레이어 해동 (Progressive Unfreezing) 적용
      - 시간에 따라 점진적으로 레이어 해동
      - `use_progressive_unfreezing` 옵션 추가
- [ ] 3. 1Cycle 학습률 스케줄러 적용
      - warmup → max LR → decay 사이클
- [ ] 4. Weight Decay 강화
      - 과적합 방지

## 진행 중
- [ ] BNN 모델 하이퍼파라미터 최적화 과정 더블체크
- [ ] 온라인러닝(OL) 파라미터도 하이퍼파라미터 최적화에 포함

## 완료
- [x] BNN에 Online Learning (OL) 적용 (OnlineTransferLearningBNN 클래스)
- [x] LOOCV epoch 수 원래대로 복원 (1/3 감소 제거)
- [x] LOOCV를 Finetune(HIFI)에만 적용하도록 수정
- [x] Scale Mixture Prior 적용 (BNN)
- [x] DNGO-OL, BNN-OL model comparison 실험 제출

## 보류/검토 필요
- [ ] Freeze/Unfreeze ratio 효과 재검토 (대규모 실험 결과 트렌드 불명확)

---
Last updated: 2024-12-04
