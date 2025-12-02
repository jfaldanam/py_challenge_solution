import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PredictForm } from './predict-form';

describe('PredictForm', () => {
  let component: PredictForm;
  let fixture: ComponentFixture<PredictForm>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PredictForm]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PredictForm);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
