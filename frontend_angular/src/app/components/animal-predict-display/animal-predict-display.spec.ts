import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AnimalPredictDisplay } from './animal-predict-display';

describe('AnimalPredictDisplay', () => {
  let component: AnimalPredictDisplay;
  let fixture: ComponentFixture<AnimalPredictDisplay>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AnimalPredictDisplay]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AnimalPredictDisplay);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
