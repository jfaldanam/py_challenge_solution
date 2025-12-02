import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TrainForm } from './train-form';

describe('TrainForm', () => {
  let component: TrainForm;
  let fixture: ComponentFixture<TrainForm>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TrainForm]
    })
    .compileComponents();

    fixture = TestBed.createComponent(TrainForm);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
